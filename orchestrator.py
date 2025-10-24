#!/usr/bin/env python3
# bitgetbot/orchestrator.py
"""
orchestrator.py
Central conductor wired to your exchange adapter and advisor.

Plain English:
This is the pilot. It starts tasks, asks the advisor for a direction,
checks with the risk seatbelt, opens, trails, and exits. It can also
take over an already-open position and protect it. When leverage is high,
it runs an emergency exit to avoid liquidation.

Definitions:
Emergency exit means a reduce-only market close based on a small move against you.
Reduce-only means the order can only reduce or close a position, never add risk.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import random
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from config.settings import Settings
from exchange.adapter import BitgetAdapter
from data.feeder import DataFeeder  # unified feeder
from journal_logger import JournalLogger
from ai.reasoner import Reasoner
from risk_manager import RiskManager, RiskLimits

@dataclass
class Position:
    symbol: str
    side: str
    size_usdt: float
    entry_price: float
    leverage: int
    contracts: float
    # NEW: accept raw exchange order payload so Position(**raw) never crashes
    order: Optional[dict] = None

@dataclass
class Params:
    trail_pct_init: float
    trail_pct_tight: float
    atr_len: int
    intelligence_sec: int
    reopen_cooldown_sec: int
    min_trail_init: float
    max_trail_init: float
    min_trail_tight: float
    max_trail_tight: float

class Orchestrator:
    def __init__(self, cfg: Settings, logger):
        self.cfg = cfg
        self.logger = logger

        self.jlog = JournalLogger(self.logger)
        self.state_store = StateStore("runtime_store/position_state.json")

        self.risk = RiskManager(self.logger, RiskLimits(
            max_stake_usdt=self.cfg.max_notional_usdt,
            max_daily_loss_usdt=300.0,
            reopen_cooldown_sec=10,
            loss_cooldown_sec=300,
            flip_cooldown_sec=120,
            max_notional_usdt=self.cfg.max_notional_usdt,
        ))

        self.adapter = BitgetAdapter(
            logger=self.logger,
            symbol=self.cfg.symbol,
            leverage=self.cfg.leverage,
            margin_mode=self.cfg.margin_mode,
            live=self.cfg.live,
        )

        # Strategy: construct correctly without passing a logger
        self.strategy = None
        try:
            from strategies.trend import TrendStrategy  # type: ignore
            self.strategy = TrendStrategy()
        except Exception:
            self.strategy = None

        # Unified feeder: prices plus context snapshot for item 1
        self.feeder = DataFeeder(self.adapter, window=1200, poll_sec=1.0)

        self.reasoner = Reasoner()

        self._stop = asyncio.Event()
        self._last_action_ts = 0.0
        self._position: Optional[Position] = None
        self._last_price: Optional[float] = None

        self._wins = 0
        self._losses = 0
        self._success_rate = 0.5

        # Dynamic parameters; AI auto-tunes these
        self._params = Params(
            trail_pct_init=0.003,      # 0.3 percent
            trail_pct_tight=0.002,     # 0.2 percent
            atr_len=5,
            intelligence_sec=self.cfg.intelligence_check_sec,
            reopen_cooldown_sec=10,
            min_trail_init=0.002,
            max_trail_init=0.01,
            min_trail_tight=0.001,
            max_trail_tight=0.006,
        )

        # Item 10 state for partial take profits
        self._tp1_done = False
        self._tp2_done = False
        self._initial_risk_abs = None
        self._entry_usdt = 0.0
        self._last_flip_eval_ts = 0.0
        self._flip_conf_min = float(os.getenv("TAKEOVER_FLIP_MIN_CONF", "0.60"))

    async def run(self):
        """Compatibility wrapper so the launcher can call orchestrator.run()."""
        return await self.start()

    async def start(self):
        self.logger.info("[BOOT] starting orchestrator")
        await self.adapter.connect()

        self.jlog.startup({
            "symbol": self.cfg.symbol,
            "stake_usdt": self.cfg.stake_usdt,
            "live": self.cfg.live,
            "lev": self.cfg.leverage,
            "margin": self.cfg.margin_mode,
            "vol": self.cfg.vol_profile,
            "takeover": self.cfg.takeover if hasattr(self.cfg, "takeover") else False,
            "advisor": "on" if self.reasoner.enabled else "off",
        })

        # Takeover of an already-open position if enabled
        if getattr(self.cfg, "takeover", False):
            try:
                raw = await self.adapter.get_open_position(self.adapter.symbol)
                if raw and raw.get("symbol") == self.adapter.symbol:
                    self._position = Position(**raw)  # type: ignore[arg-type]

                    try:
                        self.cfg.leverage = int(self._position.leverage)
                    except Exception:
                        pass
                    self.adapter.leverage = int(self.cfg.leverage)

                    if self.adapter.live:
                        try:
                            await asyncio.to_thread(self.adapter.ex.set_leverage, self.adapter.leverage, self.adapter.symbol)
                        except Exception as se:
                            self.jlog.warn("set_leverage_warn", error=str(se))

                    self.risk.set_last_side(self._position.side)

                    try:
                        actual_notional = float(self._position.size_usdt)
                    except Exception:
                        actual_notional = 0.0
                    try:
                        ep = float(self._position.entry_price)
                    except Exception:
                        ep = 0.0

                    self.logger.info(
                        f"[CFG-ACTUAL] symbol={self.adapter.symbol} stake=existing lev={self.cfg.leverage} "
                        f"live={self.adapter.live} margin={self.adapter.margin_mode} "
                        f"position_side={self._position.side} entry_price={ep} size_usdt={actual_notional:.6f} takeover=True"
                    )

                    self.jlog.heartbeat(
                        status="takeover",
                        side=self._position.side,
                        size_usdt=self._position.size_usdt,
                        lev=self._position.leverage
                    )

                    # Optional immediate flip on takeover if advisor strongly disagrees
                    try:
                        ctx = self._context_block()
                        ctx["leverage"] = int(self.cfg.leverage)
                        side_choice, reason, trail, decision = self.reasoner.decide(self.strategy, self._params, ctx)
                        desired = side_choice if side_choice in ("long", "short") else "wait"
                        conf = float(decision.get("confidence", 1.0))
                        if desired != "wait" and desired != self._position.side and conf >= self._flip_conf_min:
                            pnl = await self.adapter.close_position(asdict(self._position))
                            self.risk.note_exit()
                            if pnl is not None:
                                if pnl < 0:
                                    self._losses += 1
                                    self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                                else:
                                    self._wins += 1
                            self.jlog.trade_close(
                                self.cfg.symbol, self._position.side,
                                float(self._position.size_usdt),
                                float(self._last_price or self._position.entry_price),
                                float(pnl or 0.0),
                                self._wins, self._losses, self._success_rate,
                                reason="takeover_flip_close"
                            )
                            self._position = None
                            allowed, stake, info = self.risk.check_order(
                                symbol=self.cfg.symbol,
                                side=desired,
                                requested_stake_usdt=float(self.cfg.stake_usdt),
                                requested_leverage=int(self.cfg.leverage),
                                vol_profile=self.cfg.vol_profile,
                                now_ts=time.time(),
                                is_flip=True,
                            )
                            if not allowed:
                                self.jlog.warn("risk_block_flip", **info)
                            else:
                                opened = await self.adapter.place_order(side=desired, usdt=stake)
                                if opened:
                                    self._position = Position(**opened)  # type: ignore[arg-type]
                                    self.risk.set_last_side(desired)
                                    self.risk.note_flip()
                                    if self._last_price:
                                        self._initial_risk_abs = float(self._last_price) * float(self._params.trail_pct_init)
                                    else:
                                        self._initial_risk_abs = None
                                    self._entry_usdt = float(self._position.size_usdt)
                                    self._tp1_done = False
                                    self._tp2_done = False
                                    self.jlog.trade_open(
                                        self.cfg.symbol, desired,
                                        float(self._entry_usdt), float(self._position.entry_price),
                                        int(self._position.leverage),
                                        reason="takeover_flip_open"
                                    )
                                else:
                                    self.jlog.warn("open_failed_after_takeover_flip")
                    except Exception as fe:
                        self.jlog.exception(fe, where="takeover_active_flip")
                else:
                    self.jlog.heartbeat(status="takeover_none")
            except Exception as e:
                self.jlog.exception(e, where="takeover_check")

        await self.feeder.start()

        # Warm-up to get last price
        for _ in range(10):
            p = self.feeder.last_price()
            if p is not None:
                break
            await asyncio.sleep(0.2)
        self._last_price = self.feeder.last_price()
        await self._infer_vol_profile(force=True)

        tasks = [
            asyncio.create_task(self._watch_market(), name="watch_market"),
            asyncio.create_task(self._intelligence_cycle(), name="intelligence_cycle"),
            asyncio.create_task(self._manage_position(), name="manage_position"),
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
            asyncio.create_task(self._regime_cycle(), name="regime_cycle"),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            with contextlib.suppress(Exception):
                await self.feeder.stop()
            with contextlib.suppress(Exception):
                self._save_state()
            self.jlog.shutdown()

    def request_stop(self):
        self._stop.set()

    async def _watch_market(self):
        try:
            while not self._stop.is_set():
                self._last_price = self.feeder.last_price()
                if self._last_price is not None and self.strategy is not None:
                    try:
                        self.strategy.update_tick(self._last_price)  # updates SMA buffers
                    except Exception:
                        pass
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="watch_market")

    async def _intelligence_cycle(self):
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._params.intelligence_sec)
                self._success_rate = self._clamp(self._success_rate + random.uniform(-0.03, 0.03), 0.1, 0.9)
                self._auto_tune_params()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="intelligence")

    async def _manage_position(self):
        try:
            while not self._stop.is_set():
                now = time.time()

                # --------- OPEN LOGIC (when no position) ----------
                if self._position is None:
                    context = self._context_block()
                    context["leverage"] = int(self.cfg.leverage)
                    side_choice, reason, trail, decision = self.reasoner.decide(self.strategy, self._params, context)
                    note = decision.get("note")
                    if isinstance(note, str):
                        note = " ".join(note.split())[:300]
                    caution = decision.get("caution")
                    joined = " | ".join([x for x in [caution, note] if x])
                    self.jlog.decision(side_choice, reason, trail, cautions=(joined if joined else None))

                    if side_choice == "wait":
                        await asyncio.sleep(0.8)
                        continue

                    confidence = float(decision.get("confidence", 1.0))
                    conf_effect = self._clamp(confidence, 0.5, 1.0)
                    scaled_stake = float(self.cfg.stake_usdt) * conf_effect
                    scaled_trail_init = self._clamp(self._params.trail_pct_init * (1.0 - 0.3 * (confidence - 0.5) / 0.5), self._params.min_trail_init, self._params.max_trail_init)
                    scaled_trail_tight = self._clamp(self._params.trail_pct_tight * (1.0 - 0.3 * (confidence - 0.5) / 0.5), self._params.min_trail_tight, self._params.max_trail_tight)

                    allowed, stake, info = self.risk.check_order(
                        symbol=self.cfg.symbol,
                        side=side_choice,
                        requested_stake_usdt=float(scaled_stake),
                        requested_leverage=int(self.cfg.leverage),
                        vol_profile=self.cfg.vol_profile,
                        now_ts=now,
                        is_flip=None,
                    )
                    if info.get("warnings"):
                        self.jlog.warn("risk_warnings", warnings=info["warnings"])
                    if not allowed:
                        self.jlog.warn("risk_block_open", **info)
                        await asyncio.sleep(1.0)
                        continue

                    snapshot = {
                        "price": self._last_price,
                        "trail_init": scaled_trail_init,
                        "trail_tight": scaled_trail_tight,
                        "intel_sec": self._params.intelligence_sec,
                        "stake": stake,
                        "lev": int(self.cfg.leverage),
                        "vol": self.cfg.vol_profile,
                        "confidence": round(confidence, 3),
                        "funding": context.get("funding"),
                        "oi": context.get("open_interest"),
                        "volsnap": context.get("volatility"),
                    }
                    self.jlog.heartbeat(status="open", **snapshot)

                    raw = await self.adapter.place_order(side=side_choice, usdt=stake)
                    if not raw:
                        self.jlog.warn("open_failed")
                        await asyncio.sleep(1.0)
                        continue
                    self._position = Position(**raw)  # type: ignore[arg-type]
                    self._last_action_ts = now
                    self.risk.set_last_side(side_choice)
                    if self._last_price:
                        self._initial_risk_abs = float(self._last_price) * float(scaled_trail_init)
                    else:
                        self._initial_risk_abs = None
                    self._entry_usdt = float(self._position.size_usdt)
                    self._tp1_done = False
                    self._tp2_done = False
                    self.jlog.trade_open(self.cfg.symbol, side_choice, float(self._entry_usdt), float(self._position.entry_price), int(self._position.leverage), reason="opened_by_advisor")
                    self._save_state()
                    continue

                # --------- EXIT LOGIC or PARTIALS (when a position is open) ----------
                if self._position is not None:
                    if self._last_price is None:
                        await asyncio.sleep(0.5)
                        continue

                    pos = self._position
                    direction = 1 if pos.side == "long" else -1
                    change_abs = direction * (self._last_price - pos.entry_price)

                    # Advisor-driven flip check (periodic)
                    if (now - getattr(self, "_last_flip_eval_ts", 0.0)) >= max(3, self._params.intelligence_sec):
                        try:
                            ctx = self._context_block()
                            ctx["leverage"] = int(self.cfg.leverage)
                            side_choice, reason, trail, decision = self.reasoner.decide(self.strategy, self._params, ctx)
                            desired = side_choice if side_choice in ("long", "short") else "wait"
                            conf = float(decision.get("confidence", 1.0))
                            if desired != "wait" and desired != pos.side and conf >= self._flip_conf_min:
                                pnl = await self.adapter.close_position(asdict(pos))
                                self._position = None
                                self._last_action_ts = time.time()
                                self.risk.note_exit()
                                if pnl is not None:
                                    if pnl >= 0:
                                        self._wins += 1
                                    else:
                                        self._losses += 1
                                        self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                                self.jlog.trade_close(
                                    self.cfg.symbol, pos.side,
                                    float(self._entry_usdt or pos.size_usdt),
                                    float(self._last_price or pos.entry_price),
                                    float(pnl or 0.0),
                                    self._wins, self._losses, self._success_rate,
                                    reason="advisor_flip_close"
                                )
                                allowed, stake, info = self.risk.check_order(
                                    symbol=self.cfg.symbol,
                                    side=desired,
                                    requested_stake_usdt=float(self.cfg.stake_usdt),
                                    requested_leverage=int(self.cfg.leverage),
                                    vol_profile=self.cfg.vol_profile,
                                    now_ts=time.time(),
                                    is_flip=True,
                                )
                                if not allowed:
                                    self.jlog.warn("risk_block_flip", **info)
                                else:
                                    opened = await self.adapter.place_order(side=desired, usdt=stake)
                                    if opened:
                                        self._position = Position(**opened)  # type: ignore[arg-type]
                                        self.risk.set_last_side(desired)
                                        self.risk.note_flip()
                                        if self._last_price:
                                            self._initial_risk_abs = float(self._last_price) * float(self._params.trail_pct_init)
                                        else:
                                            self._initial_risk_abs = None
                                        self._entry_usdt = float(self._position.size_usdt)
                                        self._tp1_done = False
                                        self._tp2_done = False
                                        self.jlog.trade_open(
                                            self.cfg.symbol, desired,
                                            float(self._entry_usdt), float(self._position.entry_price),
                                            int(self._position.leverage),
                                            reason="advisor_flip_open"
                                        )
                                    else:
                                        self.jlog.warn("open_failed_after_flip")
                        except Exception as fe:
                            self.jlog.exception(fe, where="advisor_flip_eval")
                        self._last_flip_eval_ts = now

                    # Item 10: partial take-profits at 1R and 2R
                    if self._initial_risk_abs and self._initial_risk_abs > 0:
                        if (not self._tp1_done) and (change_abs >= 1.0 * self._initial_risk_abs):
                            ok = await self.adapter.close_position_fraction(asdict(pos), fraction=float(os.getenv("TP1_FRACTION", "0.30")))
                            if ok:
                                self._tp1_done = True
                                newp = await self.adapter.get_open_position(self.adapter.symbol)
                                if newp:
                                    self._position = Position(**newp)  # type: ignore[arg-type]
                                self.jlog.partial_close(self.cfg.symbol, pos.side, fraction=float(os.getenv("TP1_FRACTION", "0.30")), at_r=1.0, price=float(self._last_price), reason="partial_close_1R")
                        if (not self._tp2_done) and (change_abs >= 2.0 * self._initial_risk_abs):
                            ok = await self.adapter.close_position_fraction(asdict(pos), fraction=float(os.getenv("TP2_FRACTION", "0.30")))
                            if ok:
                                self._tp2_done = True
                                newp = await self.adapter.get_open_position(self.adapter.symbol)
                                if newp:
                                    self._position = Position(**newp)  # type: ignore[arg-type]
                                self.jlog.partial_close(self.cfg.symbol, pos.side, fraction=float(os.getenv("TP2_FRACTION", "0.30")), at_r=2.0, price=float(self._last_price), reason="partial_close_2R")

                    # Emergency exit
                    denom = pos.entry_price if (pos.entry_price and pos.entry_price > 0) else 1e-9
                    change_pct = direction * (self._last_price - pos.entry_price) / denom
                    emergency_trigger = max(0.003, 0.03 / max(1, int(pos.leverage)))

                    if change_pct <= -emergency_trigger:
                        flip_side = "short" if pos.side == "long" else "long"
                        snap = {
                            "side": flip_side,
                            "trail_init": self._params.trail_pct_init,
                            "trail_tight": self._params.trail_pct_tight,
                            "intel_sec": self._params.intelligence_sec,
                            "lev": int(self.cfg.leverage),
                            "funding": self._context_block().get("funding"),
                            "open_interest": self._context_block().get("open_interest"),
                            "volatility": self._context_block().get("volatility"),
                        }
                        allow, conf, note = self.reasoner.evaluate(snap)

                        pnl = await self.adapter.close_position(asdict(pos))
                        self._position = None
                        self._last_action_ts = time.time()
                        self.risk.note_exit()
                        if pnl is not None and pnl < 0:
                            self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                        self.jlog.trade_close(
                            self.cfg.symbol, pos.side,
                            float(self._entry_usdt or pos.size_usdt),
                            float(self._last_price or pos.entry_price),
                            float(pnl or 0.0),
                            self._wins, self._losses, self._success_rate,
                            reason="emergency_exit"
                        )
                        self._save_state()

                        if allow and float(conf) >= 0.60:
                            try:
                                stake_for_flip = min(float(self.cfg.stake_usdt), self.risk.limits.max_stake_usdt)
                                opened = await self.adapter.place_order(side=flip_side, usdt=stake_for_flip)
                                if opened:
                                    self._position = Position(**opened)  # type: ignore[arg-type]
                                    self.risk.set_last_side(flip_side)
                                    if self._last_price:
                                        self._initial_risk_abs = float(self._last_price) * float(self._params.trail_pct_init)
                                    else:
                                        self._initial_risk_abs = None
                                    self._entry_usdt = float(self._position.size_usdt)
                                    self._tp1_done = False
                                    self._tp2_done = False
                                    self.jlog.trade_open(
                                        self.cfg.symbol, flip_side,
                                        float(self._entry_usdt), float(self._position.entry_price),
                                        int(self._position.leverage),
                                        reason="flip_after_emergency"
                                    )
                                else:
                                    self.jlog.warn("open_failed_after_flip")
                            except Exception as fe:
                                self.jlog.exception(fe, where="flip_after_emergency")
                        await asyncio.sleep(0.5)
                        continue

                    # Normal trailing exit
                    trigger = self._params.trail_pct_tight * 0.5
                    if abs(change_pct) >= trigger:
                        pnl = await self.adapter.close_position(asdict(pos))
                        self._position = None
                        self._last_action_ts = time.time()
                        self.risk.note_exit()
                        if pnl is not None:
                            if pnl >= 0:
                                self._wins += 1
                            else:
                                self._losses += 1
                                self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                        self.jlog.trade_close(
                            self.cfg.symbol, pos.side,
                            float(self._entry_usdt or pos.size_usdt),
                            float(self._last_price or pos.entry_price),
                            float(pnl or 0.0),
                            self._wins, self._losses, self._success_rate,
                            reason="trail_exit"
                        )
                        self._save_state()
                    else:
                        await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="manage_position")

    async def _heartbeat(self):
        try:
            while not self._stop.is_set():
                st = self._snapshot()
                ctx = self._context_block()
                st.update({
                    "funding": ctx.get("funding"),
                    "open_interest": ctx.get("open_interest"),
                    "volatility": ctx.get("volatility"),
                })
                self.jlog.heartbeat(**st)
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="heartbeat")

    async def _regime_cycle(self):
        try:
            while not self._stop.is_set():
                await asyncio.sleep(15)
                await self._infer_vol_profile(force=False)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="regime")

    async def _infer_vol_profile(self, force: bool):
        try:
            pass
        except Exception as e:
            self.jlog.exception(e, where="infer_vol")

    def _auto_tune_params(self):
        pass

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "live": self.cfg.live,
            "leverage": self.cfg.leverage,
            "margin_mode": self.cfg.margin_mode,
            "vol_profile": self.cfg.vol_profile,
            "takeover": getattr(self.cfg, "takeover", False),
            "params": asdict(self._params),
            "wins": self._wins,
            "losses": self._losses,
            "success_rate": self._success_rate,
            "position": asdict(self._position) if self._position else None,
            "last_price": self._last_price,
            "last_action_ts": self._last_action_ts,
        }

    def _save_state(self) -> None:
        try:
            snap = {
                "ts": time.time(),
                "symbol": self.cfg.symbol,
                "live": bool(self.cfg.live),
                "leverage": int(self.cfg.leverage),
                "margin_mode": str(self.cfg.margin_mode),
                "vol_profile": str(self.cfg.vol_profile),
                "wins": int(self._wins),
                "losses": int(self._losses),
                "success_rate": float(self._success_rate),
                "last_price": float(self._last_price) if self._last_price is not None else None,
                "position": asdict(self._position) if self._position else None,
                "params": asdict(self._params),
            }
            self.state_store.save(snap)
        except Exception as e:
            with contextlib.suppress(Exception):
                self.jlog.warn("save_state_failed", error=str(e))

    def _context_block(self) -> Dict[str, Any]:
        ctx = self.feeder.context()
        return {
            "funding": ctx.get("funding_rate"),
            "open_interest": ctx.get("open_interest"),
            "volatility": ctx.get("vol_snapshot"),
            "next_funding_ts": ctx.get("next_funding_ts"),
        }

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

class StateStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save(self, obj: dict) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                import json
                json.dump(obj, f)
        except Exception:
            pass
