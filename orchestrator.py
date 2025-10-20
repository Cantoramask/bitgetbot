#!/usr/bin/env python3
"""
orchestrator.py
Central conductor for the Bitget bot, fully integrated.

Plain-English summary:
This is the pilot. It starts background tasks, keeps them alive, auto-tunes
settings gently inside safety fences, asks the RiskManager before opening a trade,
writes a clean audit trail through JournalLogger, and saves a tiny state snapshot
so it can pick up where it left off.

You can run via app.py. In paper mode this uses a placeholder exchange adapter
so you can see heartbeats and journal lines right now. In live mode the placeholder
won't auto-open positions. When your real Bitget adapter is ready, swap it in
without changing the orchestrator: just keep the same method names on the adapter.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import logging
import random
import contextlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Project modules
from state_store import StateStore, Journal
from journal_logger import JournalLogger
from risk_manager import RiskManager, RiskLimits


# ---------------------------
# Small data models
# ---------------------------
@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    size_usdt: float
    entry_price: float
    leverage: int


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


# ---------------------------
# Placeholder exchange adapter
# Replace with the real Bitget adapter when ready.
# ---------------------------
class ExchangeAdapter:
    def __init__(self, logger: logging.Logger, symbol: str, leverage: int, margin_mode: str, live: bool):
        self.logger = logger
        self.symbol = symbol
        self.live = live
        self.leverage = leverage
        self.margin_mode = margin_mode

    async def connect(self):
        self.logger.info(f"[EX] adapter ready live={self.live} lev={self.leverage} margin={self.margin_mode}")

    async def get_open_position(self) -> Optional[Position]:
        # Return None in this placeholder.
        return None

    async def fetch_ticker(self) -> Dict[str, Any]:
        price = 100_000 + random.uniform(-60, 60)
        return {"symbol": self.symbol, "price": price, "ts": time.time()}

    async def place_order(self, side: str, usdt: float) -> Optional[Position]:
        t = await self.fetch_ticker()
        pos = Position(symbol=self.symbol, side=side, size_usdt=float(usdt), entry_price=float(t["price"]), leverage=self.leverage)
        return pos

    async def close_position(self, pos: Position) -> Optional[float]:
        t = await self.fetch_ticker()
        px = float(t["price"])
        direction = 1 if pos.side == "long" else -1
        pnl_pct = direction * (px - pos.entry_price) / max(1.0, pos.entry_price)
        pnl_usdt = pos.size_usdt * pnl_pct
        return pnl_usdt


# ---------------------------
# Orchestrator
# ---------------------------
class Orchestrator:
    def __init__(self, cfg, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

        # Shared persistence and structured logging
        self.state_store = StateStore("orchestrator")
        self.events = Journal("events")
        self.jlog = JournalLogger(self.logger, self.events)

        # Risk manager with sensible defaults; tune later as needed
        self.risk = RiskManager(self.logger, RiskLimits())

        # Exchange adapter
        self.adapter = ExchangeAdapter(
            logger=self.logger,
            symbol=self.cfg.symbol,
            leverage=self.cfg.leverage,
            margin_mode=self.cfg.margin_mode,
            live=self.cfg.live,
        )

        # Live variables
        self._stop = asyncio.Event()
        self._last_action_ts = 0.0
        self._position: Optional[Position] = None
        self._params = self._make_params_from_vol_profile(self.cfg.vol_profile)

        # Rolling stats used by the auto-tuner
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
        })

        if self.cfg.takeover:
            try:
                pos = await self.adapter.get_open_position()
                if pos and pos.symbol == self.cfg.symbol:
                    self._position = pos
                    self.risk.set_last_side(pos.side)
                    self.jlog.heartbeat(status="takeover", side=pos.side, size_usdt=pos.size_usdt, lev=pos.leverage)
                else:
                    self.jlog.heartbeat(status="takeover_none")
            except Exception as e:
                self.jlog.exception(e, where="takeover_check")

        tasks = [
            asyncio.create_task(self._watch_market(), name="watch_market"),
            asyncio.create_task(self._intelligence_cycle(), name="intelligence_cycle"),
            asyncio.create_task(self._manage_position(), name="manage_position"),
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
        ]

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            self._save_state()
            self.jlog.shutdown()

    def request_stop(self):
        self._stop.set()

    # ---------------------------
    # Param logic and auto-tune
    # ---------------------------
    def _make_params_from_vol_profile(self, vol: str) -> Params:
        v = (vol or "Medium").lower()
        if v == "high":
            base = Params(
                trail_pct_init=0.006, trail_pct_tight=0.003, atr_len=10,
                intelligence_sec=max(5, getattr(self.cfg, "intelligence_check_sec", 10)),
                reopen_cooldown_sec=20,
                min_trail_init=0.002, max_trail_init=0.02,
                min_trail_tight=0.001, max_trail_tight=0.01,
                min_atr_len=5, max_atr_len=50,
                min_intel_sec=3, max_intel_sec=30,
                min_cooldown=5, max_cooldown=180,
            )
        elif v == "low":
            base = Params(
                trail_pct_init=0.010, trail_pct_tight=0.006, atr_len=20,
                intelligence_sec=max(8, getattr(self.cfg, "intelligence_check_sec", 10)),
                reopen_cooldown_sec=40,
                min_trail_init=0.002, max_trail_init=0.03,
                min_trail_tight=0.001, max_trail_tight=0.02,
                min_atr_len=5, max_atr_len=50,
                min_intel_sec=3, max_intel_sec=30,
                min_cooldown=5, max_cooldown=180,
            )
        else:
            base = Params(
                trail_pct_init=0.008, trail_pct_tight=0.004, atr_len=14,
                intelligence_sec=getattr(self.cfg, "intelligence_check_sec", 10),
                reopen_cooldown_sec=30,
                min_trail_init=0.002, max_trail_init=0.025,
                min_trail_tight=0.001, max_trail_tight=0.015,
                min_atr_len=5, max_atr_len=50,
                min_intel_sec=3, max_intel_sec=30,
                min_cooldown=5, max_cooldown=180,
            )
        return base

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

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

    # ---------------------------
    # Tasks
    # ---------------------------
    async def _watch_market(self):
        try:
            while not self._stop.is_set():
                t = await self.adapter.fetch_ticker()
                self._last_price = float(t["price"])
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.jlog.exception(e, where="watch_market")

    async def _intelligence_cycle(self):
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._params.intelligence_sec)
                # Synthetic success drift so you can see auto-tuning work today.
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

                # In paper mode, open a tiny toy position to drive the loop.
                if self._position is None and not self.cfg.live:
                    side = random.choice(["long", "short"])
                    allowed, stake, info = self.risk.check_order(
                        symbol=self.cfg.symbol,
                        side=side,
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

                    pos = await self.adapter.place_order(side=side, usdt=stake)
                    self._position = pos
                    self._last_action_ts = now
                    self.risk.set_last_side(side)
                    self.jlog.trade_open(self.cfg.symbol, side, stake, pos.entry_price, pos.leverage)
                    self._save_state()
                    continue

                if self._position is not None:
                    if self._last_price is None:
                        await asyncio.sleep(0.5)
                        continue

                    pos = self._position
                    direction = 1 if pos.side == "long" else -1
                    change_pct = direction * (self._last_price - pos.entry_price) / max(1.0, pos.entry_price)
                    trigger = self._params.trail_pct_tight * 0.5

                    if abs(change_pct) >= trigger:
                        pnl = await self.adapter.close_position(pos)
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

    # ---------------------------
    # Helpers
    # ---------------------------
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
