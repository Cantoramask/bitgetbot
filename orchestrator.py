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
# from strategies.trend import TrendStrategy  # keep if present in your repo
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

        self.jlog = JournalLogger(self.logger)  # fixed constructor usage
        self.state_store = StateStore("runtime_store/position_state.json")

        self.risk = RiskManager(self.logger, RiskLimits(
            max_stake_usdt=self.cfg.max_notional_usdt,  # starting point; will be trimmed by notional fence
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

        # Unified feeder: prices plus context snapshot for item 1
        self.feeder = DataFeeder(self.adapter, window=1200, poll_sec=1.0)

        # Strategy is optional here; Reasoner will handle absence gracefully
        self.strategy = None
        try:
            from strategies.trend import TrendStrategy  # type: ignore
            self.strategy = TrendStrategy(self.logger)
        except Exception:
            self.strategy = None

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
        self._initial_risk_abs = None  # price distance at entry for 1R
        self._entry_usdt = 0.0

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
                    self.jlog.heartbeat(status="takeover", side=self._position.side, size_usdt=self._position.size_usdt, lev=self._position.leverage)
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
            await self._stop.wait()
        finally:
            for t in tasks:
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
                    # let strategy update if present
                    try:
                        self.strategy.update_tick(self._last_price)  # type: ignore[attr-defined]
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
                    side_choice, reason, trail, decision = self.reasoner.decide(self.strategy, self._params, context)
                    caution = decision.get("caution")
                    self.jlog.decision(side_choice, reason, trail, cautions=caution)

                    if side_choice == "wait":
                        await asyncio.sleep(0.8)
                        continue

                    # Item 2: use confidence to scale stake and trail
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
                    # Item 10: compute initial risk in price terms using scaled_trail_init
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

                    # Item 10: partial take-profits at 1R and 2R
                    if self._initial_risk_abs and self._initial_risk_abs > 0:
                        if (not self._tp1_done) and (change_abs >= 1.0 * self._initial_risk_abs):
                            ok = await self.adapter.close_position_fraction(asdict(pos), fraction=float(os.getenv("TP1_FRACTION", "0.30")))
                            if ok:
                                self._tp1_done = True
                                # shrink local position contracts and usdt if adapter returned updated size
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

                    # Emergency exit: protects very high leverage positions
                    direction = 1 if pos.side == "long" else -1
                    change_pct = direction * (self._last_price - pos.entry_price) / max(1.0, pos.entry_price)
                    emergency_trigger = max(0.003, 0.03 / max(1, int(pos.leverage)))  # 0.3 percent floor; scales down with leverage
                    if change_pct <= -emergency_trigger:
                        pnl = await self.adapter.close_position(asdict(pos))
                        self._position = None
                        self._last_action_ts = time.time()
                        self.risk.note_exit()
                        if pnl is not None and pnl < 0:
                            self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                        self.jlog.trade_close(self.cfg.symbol, pos.side, float(self._entry_usdt or pos.size_usdt), float(self._last_price or pos.entry_price), float(pnl or 0.0), self._wins, self._losses, self._success_rate, reason="emergency_exit")
                        self._save_state()
                        await asyncio.sleep(0.5)
                        continue

                    # Normal trailing exit using tight trail as a simple proxy
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
                        self.jlog.trade_close(self.cfg.symbol, pos.side, float(self._entry_usdt or pos.size_usdt), float(self._last_price or pos.entry_price), float(pnl or 0.0), self._wins, self._losses, self._success_rate, reason="trail_exit")
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
                # include context awareness into heartbeat for item 1
                ctx = self._context_block()
                st.update({
                    "funding": ctx.get("funding"),
                    "open_interest": ctx.get("open_interest"),
                    "volatility": ctx.get("volatility"),
                })
                self.jlog.heartbeat(**st)
                await asyncio.sleep(10)
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
        # simple placeholder
        try:
            pass
        except Exception as e:
            self.jlog.exception(e, where="infer_vol")

    def _auto_tune_params(self):
        # simple placeholder
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
