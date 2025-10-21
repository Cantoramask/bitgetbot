
#!/usr/bin/env python3
# bitgetbot/orchestrator.py
"""
orchestrator.py
Central conductor wired to your exchange adapter and advisor in the paths defined by your project structure doc.

Plain English.
This is the pilot. It starts tasks, tunes settings inside fences, checks risk, asks the advisor if enabled, writes a clean audit trail, and saves a tiny state so it can continue smoothly. It now imports exchange/adapter.py and ai/reasoner.py so the only setup you need is your .env.
"""

from __future__ import annotations

import asyncio
import time
import random
import contextlib
from dataclasses import dataclass, asdict
from typing import Optional

from state_store import StateStore, Journal
from journal_logger import JournalLogger
from risk_manager import RiskManager, RiskLimits
from exchange.adapter import BitgetAdapter
from ai.reasoner import Reasoner
from data.feeder import DataFeeder
from strategies.trend import TrendStrategy

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
    min_atr_len: int
    max_atr_len: int
    min_intel_sec: int
    max_intel_sec: int
    min_cooldown: int
    max_cooldown: int


class Orchestrator:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        self.state_store = StateStore("orchestrator")
        self.events = Journal("events")
        self.jlog = JournalLogger(self.logger, self.events)
        self.risk = RiskManager(self.logger, RiskLimits())
        self.reasoner = Reasoner()

        self.adapter = BitgetAdapter(
            logger=self.logger,
            symbol=self.cfg.symbol,
            leverage=self.cfg.leverage,
            margin_mode=self.cfg.margin_mode,
            live=self.cfg.live,
        )

        self._stop = asyncio.Event()
        self._last_action_ts = 0.0
        self._position: Optional[Position] = None

        self._params = self._make_params_from_vol_profile(self.cfg.vol_profile)

        self.feeder = DataFeeder(self.adapter, window=1200, poll_sec=1.0)
        self.strategy = TrendStrategy(
            fast=20,
            slow=50,
            atr_len=self._params.atr_len,
            min_trail=self._params.min_trail_init,
            max_trail=self._params.max_trail_init
        )
        # tighten initial params to leverage immediately
        self._apply_leverage_tighten(int(self.cfg.leverage))

        self._vol_bucket = (self.cfg.vol_profile or "auto")
        self._last_vol_change_ts = 0.0

        self._wins = 0
        self._losses = 0
        self._success_rate = 0.5
        self._last_price: Optional[float] = None

    async def run(self):
        await self.adapter.connect()
        self.jlog.startup({
            "symbol": self.cfg.symbol,
            "live": self.cfg.live,
            "leverage": self.cfg.leverage,
            "margin": self.cfg.margin_mode,
            "vol": self.cfg.vol_profile,
            "takeover": self.cfg.takeover,
            "advisor": "on" if self.reasoner.enabled else "off",
        })

        if self.cfg.takeover:
            try:
                raw = await self.adapter.get_open_position()
                if raw and raw.get("symbol") == self.cfg.symbol:
                    self._position = Position(**raw)  # type: ignore[arg-type]
                    self.risk.set_last_side(self._position.side)
                    self.jlog.heartbeat(status="takeover", side=self._position.side, size_usdt=self._position.size_usdt, lev=self._position.leverage)
                else:
                    self.jlog.heartbeat(status="takeover_none")
            except Exception as e:
                self.jlog.exception(e, where="takeover_check")

        await self.feeder.start()

        # quick warm-up for regime inference
        for _ in range(10):
            if self.feeder.last_price() is not None:
                break
            await asyncio.sleep(0.2)
        self.logger.info("[DBG] feeder warm-up last_price=%s", str(self.feeder.last_price()))
        await self._infer_vol_profile(force=True)
        self.logger.info("[DBG] regime after warm-up vol=%s params=%s", self.cfg.vol_profile, asdict(self._params))

        tasks = [
            asyncio.create_task(self._watch_market(), name="watch_market"),
            asyncio.create_task(self._intelligence_cycle(), name="intelligence_cycle"),
            asyncio.create_task(self._manage_position(), name="manage_position"),
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
            asyncio.create_task(self._regime_cycle(), name="regime_cycle"),
        ]
        self.logger.info("[DBG] tasks started: %s", [t.get_name() for t in tasks])

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            with contextlib.suppress(Exception):
                await self.feeder.stop()
            self._save_state()
            self.jlog.shutdown()

    def request_stop(self):
        self._stop.set()

    def _make_params_from_vol_profile(self, vol: str) -> Params:
        v = (vol or "Medium").lower()
        if v == "high":
            base = Params(0.006, 0.003, 10, max(5, getattr(self.cfg, "intelligence_check_sec", 10)), 20, 0.002, 0.02, 0.001, 0.01, 5, 50, 3, 30, 5, 180)
        elif v == "low":
            base = Params(0.010, 0.006, 20, max(8, getattr(self.cfg, "intelligence_check_sec", 10)), 40, 0.002, 0.03, 0.001, 0.02, 5, 50, 3, 30, 5, 180)
        else:
            base = Params(0.008, 0.004, 14, getattr(self.cfg, "intelligence_check_sec", 10), 30, 0.002, 0.025, 0.001, 0.015, 5, 50, 3, 30, 5, 180)
        return base

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def _apply_leverage_tighten(self, lev: int) -> None:
        l = max(1, int(lev))
        f = l ** 0.5
        self._params.trail_pct_init = self._clamp(self._params.trail_pct_init / f, self._params.min_trail_init, self._params.max_trail_init)
        self._params.trail_pct_tight = self._clamp(self._params.trail_pct_tight / f, self._params.min_trail_tight, self._params.max_trail_tight)
        self._params.atr_len = self._clamp(int(round(self._params.atr_len / f)), self._params.min_atr_len, self._params.max_atr_len)
        self._params.intelligence_sec = self._clamp(int(round(self._params.intelligence_sec / f)), self._params.min_intel_sec, self._params.max_intel_sec)
        self._params.reopen_cooldown_sec = self._clamp(int(round(self._params.reopen_cooldown_sec / (f ** 0.5))), self._params.min_cooldown, self._params.max_cooldown)
        try:
            self.feeder.set_poll_sec(max(0.2, 1.0 / f))
        except Exception:
            pass
        try:
            self.strategy.set_leverage(l)
            self.strategy.set_windows(max(5, int(20 / f)), max(10, int(50 / f)))
            self.strategy.set_atr_len(self._params.atr_len)
        except Exception:
            pass
        self.jlog.knob_change("lev_tighten_factor", round(f, 4), round(f, 4), "lev_tighten")

    async def _regime_cycle(self):
        try:
            while not self._stop.is_set():
                await self._infer_vol_profile(force=False)
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="regime_cycle")

    async def _infer_vol_profile(self, force: bool) -> None:
        atrp = self.feeder.atr_pct(180)
        if atrp is None:
            return
        now = time.time()
        old = str(self._vol_bucket)
        hi = 0.006
        lo = 0.002
        pad = 0.0005
        if old == "High":
            lo_eff = hi - pad
        elif old == "Low":
            lo_eff = lo - pad
        else:
            lo_eff = lo
        if atrp < lo_eff:
            new = "Low"
        elif atrp < hi:
            new = "Medium"
        else:
            new = "High"
        dwell_ok = (now - self._last_vol_change_ts) >= 60
        if (new != old and dwell_ok) or force:
            self._last_vol_change_ts = now
            self._vol_bucket = new
            prev_params = asdict(self._params)
            self.cfg.vol_profile = new
            self._params = self._make_params_from_vol_profile(new)
            self._apply_leverage_tighten(int(self.cfg.leverage))
            try:
                self.strategy.set_atr_len(self._params.atr_len)
                if new == "High":
                    self.strategy.set_windows(12, 30)
                elif new == "Low":
                    self.strategy.set_windows(24, 60)
                else:
                    self.strategy.set_windows(20, 50)
            except Exception:
                pass
            self.jlog.knob_change("vol_profile", old, new, f"atrp={round(atrp, 6)} prev={prev_params}")

    def _auto_tune_params(self):
        sr = self._success_rate
        drift = (sr - 0.5) * 0.2
        new_trail_init = self._params.trail_pct_init * (1.0 - drift)
        new_trail_tight = self._params.trail_pct_tight * (1.0 - drift)
        new_atr_len = int(round(self._params.atr_len * (1.0 + (-drift))))
        new_intel = int(round(self._params.intelligence_sec * (1.0 + (-drift * 0.5))))
        new_cd = int(round(self._params.reopen_cooldown_sec * (1.0 + (0.5 - sr) * 0.5)))

        self._params.trail_pct_init = self._clamp(new_trail_init, self._params.min_trail_init, self._params.max_trail_init)
        self._params.trail_pct_tight = self._clamp(new_trail_tight, self._params.min_trail_tight, self._params.max_trail_tight)
        self._params.atr_len = self._clamp(new_atr_len, self._params.min_atr_len, self._params.max_atr_len)
        self._params.intelligence_sec = self._clamp(new_intel, self._params.min_intel_sec, self._params.max_intel_sec)
        self._params.reopen_cooldown_sec = self._clamp(new_cd, self._params.min_cooldown, self._params.max_cooldown)

        self.jlog.knob_change("trail_pct_init", round(new_trail_init, 5), self._params.trail_pct_init, "auto_tune")
        self.jlog.knob_change("trail_pct_tight", round(new_trail_tight, 5), self._params.trail_pct_tight, "auto_tune")
        self.jlog.knob_change("atr_len", new_atr_len, self._params.atr_len, "auto_tune")
        self.jlog.knob_change("intelligence_sec", new_intel, self._params.intelligence_sec, "auto_tune")
        self.jlog.knob_change("reopen_cooldown_sec", new_cd, self._params.reopen_cooldown_sec, "auto_tune")

    async def _watch_market(self):
        try:
            while not self._stop.is_set():
                lp = self.feeder.last_price()
                if lp is not None:
                    self._last_price = float(lp)
                    # keep strategy fed from feeder ticks
                    self.strategy.update_tick(self._last_price)
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
            self.jlog.exception(e, where="intelligence_cycle")

    async def _manage_position(self):
        try:
            while not self._stop.is_set():
                now = time.time()
                if now - self._last_action_ts < self._params.reopen_cooldown_sec:
                    await asyncio.sleep(0.5)
                    continue

                # --------- ENTRY LOGIC (no open position) ----------
                if self._position is None:
                    if self._last_price is None:
                        await asyncio.sleep(0.5)
                        continue

                    # Ask the strategy and LOG the decision so you can see why it waits
                    decision = self.strategy.decide()
                    side_choice = str(decision.get("side", "wait"))
                    reason = str(decision.get("reason", ""))
                    trail = float(decision.get("trail_pct", self._params.trail_pct_init))
                    caution = decision.get("caution")

                    # This prints lines like:
                    # [DECIDE] side=wait trail=0.80% why=warming_up note=None
                    self.jlog.decision(side_choice, reason, trail, cautions=caution)

                    if side_choice == "wait":
                        await asyncio.sleep(0.8)
                        continue

                    allowed, stake, info = self.risk.check_order(
                        symbol=self.cfg.symbol,
                        side=side_choice,
                        requested_stake_usdt=float(self.cfg.stake_usdt),
                        requested_leverage=int(self.cfg.leverage),
                        vol_profile=self.cfg.vol_profile,
                        now_ts=now,
                        is_flip=None,
                    )
                    if not allowed:
                        self.jlog.warn("risk_block_open", **info)
                        await asyncio.sleep(1.0)
                        continue

                    snapshot = {
                        "price": self._last_price,
                        "trail_init": self._params.trail_pct_init,
                        "trail_tight": self._params.trail_pct_tight,
                        "intel_sec": self._params.intelligence_sec,
                        "stake": stake,
                        "lev": self.cfg.leverage,
                        "vol": self.cfg.vol_profile,
                        "side": side_choice,
                        "strategy": decision,
                        "effective_notional": info.get("effective_notional_usdt"),
                        "cooldowns": info.get("cooldowns"),
                        "atrp": self.feeder.atr_pct(120),
                    }
                    allow, conf, note = self.reasoner.evaluate(snapshot)
                    if not allow:
                        self.jlog.advisor(False, conf, note)
                        await asyncio.sleep(1.0)
                        continue
                    self.jlog.advisor(True, conf, note)

                    # Apply any leverage clamp from RiskManager and tighten params if changed
                    lev_to_use = int(info.get("leverage", self.cfg.leverage))
                    if lev_to_use != self.adapter.leverage:
                        self.adapter.leverage = lev_to_use
                        self._apply_leverage_tighten(lev_to_use)
                        try:
                            if self.adapter.live:
                                await asyncio.to_thread(self.adapter.ex.set_leverage, lev_to_use, self.adapter.symbol)
                        except Exception as e:
                            self.jlog.warn("set_leverage_warn", error=str(e))

                    raw = await self.adapter.place_order(side=side_choice, usdt=stake)
                    if not raw:
                        self.jlog.warn("open_failed")
                        await asyncio.sleep(1.0)
                        continue
                    self._position = Position(**raw)  # type: ignore[arg-type]
                    self._last_action_ts = now
                    self.risk.set_last_side(side_choice)
                    self.jlog.trade_open(self.cfg.symbol, side_choice, stake, self._position.entry_price, self._position.leverage)
                    self._save_state()
                    continue

                # --------- EXIT LOGIC (when a position is open) ----------
                if self._position is not None:
                    if self._last_price is None:
                        await asyncio.sleep(0.5)
                        continue

                    pos = self._position
                    direction = 1 if pos.side == "long" else -1
                    change_pct = direction * (self._last_price - pos.entry_price) / max(1.0, pos.entry_price)
                    trigger = self._params.trail_pct_tight * 0.5

                    if abs(change_pct) >= trigger:
                        pnl = await self.adapter.close_position(asdict(pos))
                        self._position = None
                        self._last_action_ts = time.time()
                        self.risk.note_exit()

                        if pnl is not None and pnl >= 0:
                            self._wins += 1
                        else:
                            self._losses += 1
                        total = max(1, self._wins + self._losses)
                        self._success_rate = self._wins / total
                        self.risk.register_fill(pnl_usdt=float(pnl or 0.0))
                        self.jlog.trade_close(self.cfg.symbol, pos.side, pos.size_usdt, float(self._last_price), float(pnl or 0.0), self._wins, self._losses, self._success_rate)
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
                pos_txt = "none" if self._position is None else f"{self._position.side} {self._position.size_usdt:.2f}USDT @ {self._position.entry_price:.2f} x{self._position.leverage}"
                self.jlog.heartbeat(
                    price=self._last_price,
                    pos=pos_txt,
                    trail_init=round(self._params.trail_pct_init, 5),
                    trail_tight=round(self._params.trail_pct_tight, 5),
                    atr=self._params.atr_len,
                    intel_sec=self._params.intelligence_sec,
                    cd=self._params.reopen_cooldown_sec,
                    sr=round(self._success_rate, 3),
                )
                self.events.rotate_keep_bytes(10_000_000)
                self._save_state()
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="heartbeat")

    def _save_state(self):
        st = {
            "symbol": self.cfg.symbol,
            "live": self.cfg.live,
            "leverage": self.cfg.leverage,
            "margin_mode": self.cfg.margin_mode,
            "vol_profile": self.cfg.vol_profile,
            "takeover": self.cfg.takeover,
            "params": asdict(self._params),
            "wins": self._wins,
            "losses": self._losses,
            "success_rate": self._success_rate,
            "position": asdict(self._position) if self._position else None,
            "last_price": self._last_price,
            "last_action_ts": self._last_action_ts,
        }
        self.state_store.save(st)
