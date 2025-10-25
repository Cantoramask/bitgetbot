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
    # accept raw exchange order payload safely
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

        # Strategy
        self.strategy = None
        try:
            from strategies.trend import TrendStrategy  # type: ignore
            self.strategy = TrendStrategy()
        except Exception:
            self.strategy = None

        # Feeder
        self.feeder = DataFeeder(self.adapter, window=1200, poll_sec=1.0)

        self.reasoner = Reasoner()

        self._stop = asyncio.Event()
        self._last_action_ts = 0.0
        self._position: Optional[Position] = None
        self._last_price: Optional[float] = None

        self._wins = 0
        self._losses = 0
        self._success_rate = 0.5

        # Loss gating (prevents instant global cooldown on tiny losses)
        self._consec_losses = 0
        self._big_loss_frac = float(os.getenv("BIG_LOSS_FRAC", "0.20"))  # 20% of stake
        self._big_loss_abs = float(os.getenv("BIG_LOSS_USDT", "0"))      # hard USD floor (optional)
        self._consec_loss_limit = int(float(os.getenv("CONSEC_LOSS_LIMIT", "3")))

        # Profit-aligned flip gating (no timers)
        self._flip_min_conf = float(os.getenv("FLIP_MIN_CONF", "0.65"))          # min advisor confidence to even consider flip
        self._flip_stability_N = int(os.getenv("FLIP_STABILITY_N", "3"))         # consecutive opposite signals required
        self._flip_min_move_pct = float(os.getenv("FLIP_MIN_MOVE_PCT", "0.0015"))# 0.15% min move from entry to consider flip
        self._tx_cost_bps = float(os.getenv("TX_COST_BPS", "6.0"))               # fees+slippage in basis points (0.06%)
        self._edge_buffer_bps = float(os.getenv("EDGE_BUFFER_BPS", "4.0"))       # extra edge above cost to justify flip

        # running vote for stability
        self._flip_vote = {"side": None, "count": 0, "last_ts": 0.0}

        # Dynamic parameters; AI auto-tunes these
        self._params = Params(
            trail_pct_init=0.003,      # 0.3%
            trail_pct_tight=0.002,     # 0.2%
            atr_len=5,
            intelligence_sec=max(8, int(self.cfg.intelligence_check_sec)),
            reopen_cooldown_sec=10,
            min_trail_init=0.002,
            max_trail_init=0.01,
            min_trail_tight=0.001,
            max_trail_tight=0.006,
        )

        # Partial TPs
        self._tp1_done = False
        self._tp2_done = False
        self._initial_risk_abs = None
        self._entry_usdt = 0.0
        self._last_flip_eval_ts = 0.0
        self._flip_conf_min = float(os.getenv("TAKEOVER_FLIP_MIN_CONF", "0.60"))

        # Grace window after takeover to suppress immediate flips/forced closes
        self._takeover_grace_sec = int(float(os.getenv("TAKEOVER_GRACE_SEC", "60")))
        self._takeover_grace_until = 0.0

        # Defaults remembered for decay toward baseline
        self._defaults = {
            "trail_pct_init": self._params.trail_pct_init,
            "trail_pct_tight": self._params.trail_pct_tight,
            "intelligence_sec": self._params.intelligence_sec,
        }

        # Regime hysteresis bands (per-tick fractional vol)
        self._regime_cut_low = 0.0015
        self._regime_cut_high = 0.005
        self._regime_hysteresis = 0.10  # 10% band to reduce flicker

    async def run(self):
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
                    self._position = Position(**raw)  # keep exchange leverage/size as-is
                    self._takeover_grace_until = time.time() + float(self._takeover_grace_sec)

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
                        f"[CFG-ACTUAL] symbol={self.adapter.symbol} stake=existing lev={self._position.leverage} "
                        f"live={self.adapter.live} margin={self.adapter.margin_mode} "
                        f"position_side={self._position.side} entry_price={ep} size_usdt={actual_notional:.6f} takeover=True"
                    )
                    self.jlog.heartbeat(
                        status="takeover",
                        side=self._position.side,
                        size_usdt=self._position.size_usdt,
                        lev=self._position.leverage
                    )
                    # No immediate flip during grace.
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
                        self.strategy.update_tick(self._last_price)
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
                self._success_rate = self._clamp(
                    self._success_rate + random.uniform(-0.03, 0.03),
                    0.1,
                    0.9,
                )
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
                    context["actual_leverage"] = int(self.cfg.leverage)
                    side_choice, reason, trail, decision = self.reasoner.decide(self.strategy, self._params, context)

                    note = decision.get("note")
                    if isinstance(note, str):
                        note = " ".join(note.split())[:300]
                    caution = decision.get("caution")
                    joined = " | ".join([x for x in [caution, note] if x])
                    self.jlog.decision(side_choice, reason, trail, cautions=(joined if joined else None))

                    # Strong gate: only open when advisor explicitly allows and is not warning.
                    if side_choice == "wait" or not bool(decision.get("allow")) or reason == "advisor_warn":
                        await asyncio.sleep(0.8)
                        continue

                    confidence = float(decision.get("confidence", 1.0))
                    conf_effect = self._clamp(confidence, 0.5, 1.0)

                    allowed, stake_usdt, info = self.risk.check_order(
                        symbol=self.cfg.symbol,
                        side=side_choice,
                        requested_stake_usdt=float(self.cfg.stake_usdt) * conf_effect,
                        requested_leverage=int(self.cfg.leverage),
                        vol_profile=self.cfg.vol_profile,
                        now_ts=time.time(),
                        is_flip=None,
                        confidence=confidence,
                    )
                    if not allowed:
                        self.jlog.warn("risk_block_open", **info)
                        await asyncio.sleep(0.8)
                        continue

                    opened = await self.adapter.place_order(side=side_choice, usdt=stake_usdt)
                    if opened:
                        self._position = Position(**opened)  # type: ignore[arg-type]
                        self.risk.set_last_side(side_choice)
                        if self._last_price:
                            self._initial_risk_abs = float(self._last_price) * float(self._params.trail_pct_init)
                        else:
                            self._initial_risk_abs = None
                        self._entry_usdt = float(self._position.size_usdt)
                        self._tp1_done = False
                        self._tp2_done = False
                        self.jlog.trade_open(
                            self.cfg.symbol, side_choice,
                            float(self._entry_usdt), float(self._position.entry_price),
                            int(self._position.leverage),
                            reason="advisor_open"
                        )
                        self._save_state()
                    else:
                        self.jlog.warn("open_failed")
                    await asyncio.sleep(0.8)
                    continue

                # --------- HAVE POSITION: manage / flip / flat / exit ----------
                pos = self._position
                direction = 1 if pos.side == "long" else -1

                # Update last price safety
                if self._last_price is None:
                    await asyncio.sleep(0.4)
                    continue

                # Advisor-driven flip check (periodic)
                if now - float(self._last_flip_eval_ts or 0.0) >= max(2.0, float(self._params.intelligence_sec)):
                    try:
                        ctx = self._context_block()
                        ctx["leverage"] = int(pos.leverage)
                        ctx["actual_leverage"] = int(pos.leverage)
                        desired, why, trail, decision = self.reasoner.decide(self.strategy, self._params, ctx)
                        conf = float(decision.get("confidence", 0.0))

                        do_flip, econ_why = self._should_flip(pos, desired, conf, ctx)
                        why = (why or "") + (f" | econ={econ_why}" if econ_why else "")

                        if do_flip:
                            # Pre-check risk BEFORE closing.
                            allowed, stake, info = self.risk.check_order(
                                symbol=self.cfg.symbol,
                                side=desired,
                                requested_stake_usdt=float(self.cfg.stake_usdt),
                                requested_leverage=int(pos.leverage),
                                vol_profile=self.cfg.vol_profile,
                                now_ts=time.time(),
                                is_flip=True,
                                confidence=conf,
                            )
                            if not allowed:
                                self.jlog.warn("risk_block_flip", gate="pre", why=why, **info)
                            else:
                                # Close current
                                pnl = await self.adapter.close_position(asdict(pos))
                                self._position = None
                                self._last_action_ts = time.time()
                                self.risk.note_exit()
                                self._note_trade_result(pnl)

                                self.jlog.trade_close(
                                    self.cfg.symbol, pos.side,
                                    float(self._entry_usdt or pos.size_usdt),
                                    float(self._last_price or pos.entry_price),
                                    float(pnl or 0.0),
                                    self._wins, self._losses, self._success_rate,
                                    reason="advisor_flip_close"
                                )

                                # Open new side
                                opened = await self.adapter.place_order(side=desired, usdt=stake)
                                if opened:
                                    self._position = Position(**opened)  # type: ignore[arg-type]
                                    try:
                                        self.risk.set_last_side(desired)
                                        self.risk.note_flip()
                                    except Exception:
                                        pass
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
                                    self._save_state()
                                else:
                                    self.jlog.warn("open_failed_after_flip", why=why)

                        else:
                            # If advisor says wait or did not allow, do not discretionary-flat on tiny edge.
                            if not bool(decision.get("allow", False)) or why.find("advisor_warn") != -1:
                                self.jlog.decision("hold", "advisor_not_allowed", trail, cautions=why)
                            elif desired == "wait":
                                self.jlog.decision("hold", "advisor_wait", trail, cautions=why)
                            else:
                                # Prefer flat over flip when edge is tiny only if re-entry wouldn't be blocked
                                denom = pos.entry_price if pos.entry_price else 1e-9
                                change_pct_signed = (1 if pos.side == "long" else -1) * (self._last_price - pos.entry_price) / denom
                                change_bps = 10000.0 * change_pct_signed
                                if abs(change_bps) <= (self._tx_cost_bps * 1.2):
                                    allowed_reopen, _, info_reopen = self.risk.check_order(
                                        symbol=self.cfg.symbol,
                                        side=pos.side,
                                        requested_stake_usdt=float(self.cfg.stake_usdt),
                                        requested_leverage=int(pos.leverage),
                                        vol_profile=self.cfg.vol_profile,
                                        now_ts=time.time(),
                                        is_flip=None,
                                        confidence=conf,
                                    )
                                    if allowed_reopen:
                                        pnl = await self.adapter.close_position(asdict(pos))
                                        self._position = None
                                        self._last_action_ts = time.time()
                                        self.risk.note_exit()
                                        self._note_trade_result(pnl)

                                        self.jlog.trade_close(
                                            self.cfg.symbol, pos.side,
                                            float(self._entry_usdt or pos.size_usdt),
                                            float(self._last_price or pos.entry_price),
                                            float(pnl or 0.0),
                                            self._wins, self._losses, self._success_rate,
                                            reason="advisor_flat_small_edge"
                                        )
                                    else:
                                        self.jlog.decision("hold", "flat_blocked_by_reopen_cooldown", trail, cautions=info_reopen.get("reasons") or why)
                                else:
                                    self.jlog.decision("hold", "flip_gated", trail, cautions=why)
                    except Exception as fe:
                        self.jlog.exception(fe, where="advisor_flip_eval")
                    self._last_flip_eval_ts = now

                # Partial take-profits at 1R and 2R
                denom = pos.entry_price if pos.entry_price else 1e-9
                change_abs = abs(self._last_price - pos.entry_price)
                if self._initial_risk_abs and self._initial_risk_abs > 0:
                    if (not self._tp1_done) and (change_abs >= 1.0 * self._initial_risk_abs):
                        ok = await self.adapter.close_position_fraction(asdict(pos), fraction=float(os.getenv("TP1_FRACTION", "0.30")))
                        if ok:
                            self._tp1_done = True
                            newp = await self.adapter.get_open_position(self.adapter.symbol)
                            if newp:
                                self._position = Position(**newp)  # type: ignore[arg-type]
                            self.jlog.partial_close(self.cfg.symbol, pos.side, fraction=float(os.getenv("TP1_FRACTION", "0.30")), at_r=1.0, price=float(self._last_price), reason="partial_close_1R")
                            self._save_state()
                    if (not self._tp2_done) and (change_abs >= 2.0 * self._initial_risk_abs):
                        ok = await self.adapter.close_position_fraction(asdict(pos), fraction=float(os.getenv("TP2_FRACTION", "0.30")))
                        if ok:
                            self._tp2_done = True
                            newp = await self.adapter.get_open_position(self.adapter.symbol)
                            if newp:
                                self._position = Position(**newp)  # type: ignore[arg-type]
                            self.jlog.partial_close(self.cfg.symbol, pos.side, fraction=float(os.getenv("TP2_FRACTION", "0.30")), at_r=2.0, price=float(self._last_price), reason="partial_close_2R")
                            self._save_state()

                # Emergency exit
                denom = pos.entry_price if (pos.entry_price and pos.entry_price > 0) else 1e-9
                change_pct = direction * (self._last_price - pos.entry_price) / denom
                emergency_trigger = max(0.003, 0.03 / max(1, int(pos.leverage)))
                if change_pct <= -emergency_trigger:
                    flip_side = "short" if pos.side == "long" else "long"
                    allow, conf, note = self.reasoner.evaluate({
                        "side": flip_side,
                        "trail_init": self._params.trail_pct_init,
                        "trail_tight": self._params.trail_pct_tight,
                        "intel_sec": self._params.intelligence_sec,
                        "lev": int(pos.leverage),
                        "funding": self._context_block().get("funding"),
                        "open_interest": self._context_block().get("open_interest"),
                        "volatility": self._context_block().get("volatility"),
                    })[:3]

                    pnl = await self.adapter.close_position(asdict(pos))
                    self._position = None
                    self._last_action_ts = time.time()
                    self.risk.note_exit()
                    self._note_trade_result(pnl)

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
                                self._save_state()
                            else:
                                self.jlog.warn("open_failed_after_flip")
                        except Exception as fe:
                            self.jlog.exception(fe, where="flip_after_emergency")
                    await asyncio.sleep(0.5)
                    continue

                # Normal trailing exit with fee guard
                trigger = self._params.trail_pct_tight * 0.5
                tx_guard_bps = self._tx_cost_bps * 1.2
                move_bps_from_entry = 10000.0 * ((self._last_price - pos.entry_price) / denom) * (1 if pos.side == "long" else -1)
                if change_pct <= -trigger and abs(move_bps_from_entry) > tx_guard_bps:
                    pnl = await self.adapter.close_position(asdict(pos))
                    self._position = None
                    self._last_action_ts = time.time()
                    self.risk.note_exit()
                    self._note_trade_result(pnl)

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
            hb_interval = int(float(os.getenv("HEARTBEAT_INTERVAL_SEC", "20")))
            while not self._stop.is_set():
                st = self._snapshot() or {}
                ctx = self._context_block() or {}
                st.update({
                    "funding": ctx.get("funding"),
                    "open_interest": ctx.get("open_interest"),
                    "volatility": ctx.get("volatility"),
                    "vol_slope": getattr(self.strategy, "vol_slope_status", None),
                })
                self.jlog.heartbeat(**st)
                await asyncio.sleep(max(1, hb_interval))
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
        """
        Map feeder's per-tick fractional volatility to a regime with small hysteresis.
        Skips updates when context is stale or during warm-up, unless force=True.
        """
        try:
            ctx = self.feeder.context()
            vol = float(ctx.get("vol_snapshot") or 0.0)
            is_stale = bool(ctx.get("is_stale_ctx"))
            tick_count = int(ctx.get("tick_count") or 0)

            if not force:
                if is_stale or tick_count < 60:
                    return

            current = str(self.cfg.vol_profile or "Medium")

            low_cut_up = self._regime_cut_low * (1.0 + self._regime_hysteresis)
            low_cut_down = self._regime_cut_low * (1.0 - self._regime_hysteresis)
            high_cut_up = self._regime_cut_high * (1.0 + self._regime_hysteresis)
            high_cut_down = self._regime_cut_high * (1.0 - self._regime_hysteresis)

            new_profile = current
            if current == "Low":
                if vol >= low_cut_up and vol < high_cut_down:
                    new_profile = "Medium"
                elif vol >= high_cut_up:
                    new_profile = "High"
            elif current == "Medium":
                if vol < low_cut_down:
                    new_profile = "Low"
                elif vol >= high_cut_up:
                    new_profile = "High"
            else:  # current == "High" or unknown
                if vol < high_cut_down and vol >= low_cut_up:
                    new_profile = "Medium"
                elif vol < low_cut_down:
                    new_profile = "Low"

            if force or new_profile != current:
                self.cfg.vol_profile = new_profile
                self.jlog.heartbeat(status="regime", vol_profile=new_profile, vol_snapshot=vol)
        except Exception as e:
            self.jlog.exception(e, where="infer_vol")

    def _auto_tune_params(self):
        """
        Light-touch tuning. Keep inside provided min and max. Drift back to defaults.
        Skip when context is stale or during warm-up.
        """
        try:
            ctx = self.feeder.context()
            is_stale = bool(ctx.get("is_stale_ctx"))
            tick_count = int(ctx.get("tick_count") or 0)
            if is_stale or tick_count < 60:
                return

            # decay toward defaults so we never ratchet forever
            decay = 0.2
            self._params.trail_pct_init = (
                (1 - decay) * self._params.trail_pct_init + decay * self._defaults["trail_pct_init"]
            )
            self._params.trail_pct_tight = (
                (1 - decay) * self._params.trail_pct_tight + decay * self._defaults["trail_pct_tight"]
            )
            self._params.intelligence_sec = int(
                round((1 - decay) * self._params.intelligence_sec + decay * self._defaults["intelligence_sec"])
            )

            regime = str(self.cfg.vol_profile or "Medium")
            sr = float(self._success_rate)

            if regime == "High" or sr < 0.45:
                self._params.trail_pct_init = min(self._params.max_trail_init, self._params.trail_pct_init * 1.10)
                self._params.trail_pct_tight = min(self._params.max_trail_tight, self._params.trail_pct_tight * 1.10)
                self._params.intelligence_sec = min(30, self._params.intelligence_sec + 1)
            elif regime == "Low" and sr > 0.60:
                self._params.trail_pct_init = max(self._params.min_trail_init, self._params.trail_pct_init * 0.95)
                self._params.trail_pct_tight = max(self._params.min_trail_tight, self._params.trail_pct_tight * 0.95)
                self._params.intelligence_sec = max(6, self._params.intelligence_sec - 1)

            # final clamps
            self._params.trail_pct_init = self._clamp(self._params.trail_pct_init, self._params.min_trail_init, self._params.max_trail_init)
            self._params.trail_pct_tight = self._clamp(self._params.trail_pct_tight, self._params.min_trail_tight, self._params.max_trail_tight)
        except Exception as e:
            self.jlog.exception(e, where="auto_tune")

    def _snapshot(self) -> Dict[str, Any]:
        # Use actual leverage in snapshot if a position is open, else configured leverage
        lev = int(self._position.leverage) if self._position else int(self.cfg.leverage)
        return {
            "live": self.cfg.live,
            "leverage": lev,
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
                "leverage": int(self._position.leverage) if self._position else int(self.cfg.leverage),
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

    # -------------------------
    # Loss gating helper
    # -------------------------
    def _note_trade_result(self, pnl: Optional[float]) -> None:
        """Updates W/L counters and only calls risk.register_fill when losses are big or consecutive."""
        pnl_val = float(pnl or 0.0)

        if pnl_val >= 0:
            self._wins += 1
            self._consec_losses = 0
            return

        # it's a loss
        self._losses += 1
        self._consec_losses += 1

        # thresholds
        stake = float(self.cfg.stake_usdt) or 0.0
        big_by_frac = abs(pnl_val) >= (stake * max(0.0, self._big_loss_frac))
        big_by_abs = (self._big_loss_abs > 0.0) and (abs(pnl_val) >= self._big_loss_abs)
        many_in_row = self._consec_losses >= max(1, self._consec_loss_limit)

        if big_by_frac or big_by_abs or many_in_row:
            # Only here do we inform RiskManager, which may trigger loss cooldowns.
            try:
                self.risk.register_fill(pnl_usdt=pnl_val)
            except Exception:
                pass

    def _estimate_edge_bps(self, conf: float, ctx: Dict[str, Any]) -> float:
        """
        Rough edge proxy in basis points.
        - Use advisor confidence above 0.5 as a multiplier of recent realized volatility snapshot (pct).
        - If no volatility available, fall back to a fixed tiny baseline.
        """
        vol_pct = float(ctx.get("volatility") or 0.002)  # 0.2% fallback
        conf_excess = max(0.0, conf - 0.5) / 0.5  # 0..1 when conf in [0.5,1]
        edge_pct = conf_excess * vol_pct                     # % move we expect to capture
        return 10000.0 * edge_pct                            # convert % to bps

    def _update_flip_vote(self, desired_side: str) -> int:
        now = time.time()
        if desired_side and self._flip_vote["side"] == desired_side:
            self._flip_vote["count"] += 1
        else:
            self._flip_vote = {"side": desired_side, "count": 1, "last_ts": now}
        self._flip_vote["last_ts"] = now
        return self._flip_vote["count"]

    def _should_flip(self, pos: "Position", desired: str, conf: float, ctx: Dict[str, Any]) -> (bool, str):
        """
        Decide if flipping is economically justified (profit-aligned).
        Returns (ok, reason).
        """
        if desired not in ("long", "short") or desired == pos.side:
            return (False, "same_or_invalid_side")

        # 1) confidence gate
        if conf < self._flip_min_conf:
            return (False, f"conf<{self._flip_min_conf}")

        # 2) stability gate (consecutive opposite recommendations)
        votes = self._update_flip_vote(desired)
        if votes < self._flip_stability_N:
            return (False, f"stability<{self._flip_stability_N} (votes={votes})")

        # 3) no-churn zone around entry
        denom = pos.entry_price if pos.entry_price else 1e-9
        move_from_entry_pct = abs((self._last_price - pos.entry_price) / denom)
        if move_from_entry_pct < self._flip_min_move_pct:
            return (False, f"move<{self._flip_min_move_pct*100:.2f}%")

        # 4) edge > cost + buffer
        edge_bps = self._estimate_edge_bps(conf, ctx)
        needed_bps = self._tx_cost_bps + self._edge_buffer_bps
        if edge_bps < needed_bps:
            return (False, f"edge_bps<{needed_bps} (edge={edge_bps:.1f})")

        return (True, f"ok edge={edge_bps:.1f}bps votes={votes}")

    # -------------------------
    # NEW: atomic close & state refresh helpers

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
