#!/usr/bin/env python3
"""
orchestrator.py
Central conductor for the Bitget bot.

What this file does in plain English:
It is the pilot. It starts and supervises the background tasks, keeps them alive,
and gently auto-tunes the bot's knobs within safe fences so nothing goes wild.
It reads your startup choices from AppConfig, including takeover and vol_profile.

Key ideas explained simply:
- Orchestrator: the boss loop that starts other loops and tells them when to stop.
- Cooldown: a waiting period after an action to avoid flip-flopping too fast.
- Fence: a safe min and max range for any auto-tuned setting so it never goes crazy.
- Journal: a small file where we append what happened and why, one line per event.
- State snapshot: the current small memory of the bot so it can resume more easily.

This file deliberately avoids importing real exchange libraries.
It ships with a simple placeholder ExchangeAdapter that pretends the market is calm.
When your adapter is ready, replace ExchangeAdapter with the real one and keep
the same method names.
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
    # Tunable internals with fences
    trail_pct_init: float
    trail_pct_tight: float
    atr_len: int
    intelligence_sec: int
    reopen_cooldown_sec: int

    # Fences for safety
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
# Lightweight journal and state store
# ---------------------------
class Journal:
    def __init__(self, folder: str = "data"):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, "journal.jsonl")

    def write(self, event: str, payload: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def rotate_keep_days(self, days: int = 60):
        # Simple size guard. We avoid time parsing to keep it tiny.
        try:
            if os.path.getsize(self.path) > 10_000_000:
                backup = self.path + ".bak"
                with open(self.path, "rb") as src, open(backup, "wb") as dst:
                    dst.write(src.read()[-5_000_000:])  # keep last ~5 MB
                os.replace(backup, self.path)
        except FileNotFoundError:
            pass


class StateStore:
    def __init__(self, folder: str = "data"):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, "state.json")

    def save(self, state: Dict[str, Any]):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}


# ---------------------------
# Placeholder exchange adapter
# Replace with your real adapter when ready.
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
        # Placeholder. Return None to indicate no existing position.
        # In your real adapter, fetch from exchange and build Position.
        return None

    async def fetch_ticker(self) -> Dict[str, Any]:
        # Placeholder market feed. Generates a stable price with tiny noise.
        price = 100_000 + random.uniform(-50, 50)
        return {"symbol": self.symbol, "price": price, "ts": time.time()}

    async def place_order(self, side: str, usdt: float) -> Optional[Position]:
        # Placeholder "order". Pretend immediate fill at current fetch_ticker price.
        t = await self.fetch_ticker()
        pos = Position(symbol=self.symbol, side=side, size_usdt=float(usdt), entry_price=float(t["price"]), leverage=self.leverage)
        self.logger.info(f"[EX] opened {side} {usdt:.2f}USDT at {pos.entry_price:.2f}")
        return pos

    async def close_position(self, pos: Position) -> Optional[float]:
        # Placeholder "close". Return pretend PnL in USDT based on 0.02 percent random move.
        t = await self.fetch_ticker()
        px = float(t["price"])
        direction = 1 if pos.side == "long" else -1
        pnl_pct = direction * (px - pos.entry_price) / max(1.0, pos.entry_price)
        pnl_usdt = pos.size_usdt * pnl_pct
        self.logger.info(f"[EX] closed {pos.side} {pos.size_usdt:.2f}USDT at {px:.2f} pnl={pnl_usdt:.2f}USDT")
        return pnl_usdt


# ---------------------------
# Orchestrator
# ---------------------------
class Orchestrator:
    def __init__(self, cfg, logger: logging.Logger):
        # AppConfig object from app.py
        self.cfg = cfg
        self.logger = logger

        # Core components
        self.journal = Journal()
        self.state = StateStore()
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
        self._success_rate = 0.5  # start neutral
        self._vol_hint = self.cfg.vol_profile  # High Medium Low

    # -------------
    # Public API
    # -------------
    async def run(self):
        self.logger.info("[ORCH] starting")
        await self.adapter.connect()
        self.journal.rotate_keep_days(60)

        if self.cfg.takeover:
            try:
                pos = await self.adapter.get_open_position()
                if pos and pos.symbol == self.cfg.symbol:
                    self._position = pos
                    self.logger.info(f"[ORCH] takeover active side={pos.side} size={pos.size_usdt:.2f} lev={pos.leverage}")
                    self.journal.write("takeover", {"symbol": pos.symbol, "side": pos.side, "size": pos.size_usdt, "lev": pos.leverage})
                else:
                    self.logger.info("[ORCH] takeover requested but no matching open position found")
            except Exception as e:
                self.logger.info(f"[ORCH] takeover check failed: {e}")

        # Launch supervised tasks
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

            # Final state save
            self._save_state()
            self.logger.info("[ORCH] stopped")

    def request_stop(self):
        self._stop.set()

    # -------------
    # Param logic
    # -------------
    def _make_params_from_vol_profile(self, vol: str) -> Params:
        v = (vol or "Medium").lower()
        if v == "high":
            base = Params(
                trail_pct_init=0.006,  # 0.6 percent
                trail_pct_tight=0.003,  # 0.3 percent
                atr_len=10,
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
                trail_pct_init=0.010,   # 1.0 percent
                trail_pct_tight=0.006,  # 0.6 percent
                atr_len=20,
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
                trail_pct_init=0.008,   # 0.8 percent
                trail_pct_tight=0.004,  # 0.4 percent
                atr_len=14,
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
        # Simple adaptive logic that nudges settings inside fences.
        # The success rate drives tightening or loosening of trails.
        sr = self._success_rate  # 0 to 1
        drift = (sr - 0.5) * 0.2  # small adjustment band

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

    # -------------
    # Tasks
    # -------------
    async def _watch_market(self):
        # In a real bot, this would subscribe to a websocket.
        # Here we fetch a dummy ticker every second and remember the last price.
        self._last_price = None
        try:
            while not self._stop.is_set():
                t = await self.adapter.fetch_ticker()
                self._last_price = float(t["price"])
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.info(f"[watch_market] error: {e}")

    async def _intelligence_cycle(self):
        # Periodically auto-tune parameters and update success stats.
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._params.intelligence_sec)
                # Update a simple synthetic success score that drifts a little.
                drift = random.uniform(-0.03, 0.03)
                self._success_rate = self._clamp(self._success_rate + drift, 0.1, 0.9)

                self._auto_tune_params()
                self.journal.write("auto_tune", {
                    "success_rate": round(self._success_rate, 3),
                    "params": {
                        "trail_init": round(self._params.trail_pct_init, 5),
                        "trail_tight": round(self._params.trail_pct_tight, 5),
                        "atr_len": self._params.atr_len,
                        "intel_sec": self._params.intelligence_sec,
                        "reopen_cd": self._params.reopen_cooldown_sec,
                    }
                })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.info(f"[intelligence_cycle] error: {e}")

    async def _manage_position(self):
        # Placeholder manager. It demonstrates cooldown and journal entries.
        try:
            while not self._stop.is_set():
                now = time.time()

                # Respect reopen cooldown
                if now - self._last_action_ts < self._params.reopen_cooldown_sec:
                    await asyncio.sleep(0.5)
                    continue

                # If no position, optionally open a tiny placeholder one in paper mode only.
                if self._position is None and not self.cfg.live:
                    # Tiny fake entry to drive the loop. In live mode we do nothing here.
                    side = random.choice(["long", "short"])
                    self._position = await self.adapter.place_order(side=side, usdt=float(self.cfg.stake_usdt))
                    self._last_action_ts = now
                    self.journal.write("open", {
                        "side": side,
                        "usdt": float(self.cfg.stake_usdt),
                        "trail_init": self._params.trail_pct_init,
                        "trail_tight": self._params.trail_pct_tight
                    })
                    continue

                # If position exists, consider closing based on a trivial trailing rule surrogate.
                if self._position is not None:
                    # Using last price if present for a toy trailing decision
                    if getattr(self, "_last_price", None) is None:
                        await asyncio.sleep(0.5)
                        continue

                    pos = self._position
                    direction = 1 if pos.side == "long" else -1
                    change_pct = direction * (self._last_price - pos.entry_price) / max(1.0, pos.entry_price)

                    # Soft trail: if price pulls back by trail_pct_tight from the best seen,
                    # we would close. Since we do not track best price here, use a simple check.
                    trigger = self._params.trail_pct_tight * 0.5
                    if abs(change_pct) >= trigger:
                        pnl = await self.adapter.close_position(pos)
                        self._position = None
                        self._last_action_ts = time.time()
                        if pnl is not None and pnl >= 0:
                            self._wins += 1
                        else:
                            self._losses += 1
                        total = max(1, self._wins + self._losses)
                        self._success_rate = self._wins / total

                        self.journal.write("close", {
                            "pnl_usdt": round(float(pnl or 0.0), 4),
                            "wins": self._wins,
                            "losses": self._losses,
                            "success_rate": round(self._success_rate, 3),
                        })
                    else:
                        # Hold and wait
                        await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.info(f"[manage_position] error: {e}")

    async def _heartbeat(self):
        try:
            while not self._stop.is_set():
                p = self._position
                pos_txt = "none" if p is None else f"{p.side} {p.size_usdt:.2f}USDT @ {p.entry_price:.2f} x{p.leverage}"
                self.logger.info(
                    "[HB] price=%s pos=%s | trail_init=%.4f trail_tight=%.4f atr=%d intel=%ds cd=%ds sr=%.2f",
                    f"{getattr(self, '_last_price', None)}",
                    pos_txt,
                    self._params.trail_pct_init,
                    self._params.trail_pct_tight,
                    self._params.atr_len,
                    self._params.intelligence_sec,
                    self._params.reopen_cooldown_sec,
                    self._success_rate,
                )
                self._save_state()
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.info(f"[heartbeat] error: {e}")

    # -------------
    # Helpers
    # -------------
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
            "last_price": getattr(self, "_last_price", None),
            "last_action_ts": self._last_action_ts,
        }
        self.state.save(st)
