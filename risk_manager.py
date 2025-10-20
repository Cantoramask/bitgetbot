#!/usr/bin/env python3
"""
risk_manager.py
Seatbelt for the Bitget bot.

Plain-English summary:
This file protects you from silly or dangerous trades. Before any order is sent,
the orchestrator asks RiskManager: "Is this safe, and if not, how should I trim it?"
RiskManager checks leverage, stake size, daily loss, and cooldowns. It can block a
trade or reduce the stake to fit the rules. After a trade closes, the orchestrator
calls `register_fill(pnl_usdt=...)` so daily loss limits stay accurate.

Simple definitions:
- Cooldown: a waiting period after an action before you are allowed to act again.
- Daily loss limit: the maximum money you are prepared to lose in a single day.
- Flip: closing a long and immediately opening a short (or the reverse).

This module stores a tiny JSON file at data/risk_state.json to remember today's
losses and the last time you flipped or took a loss, so cooldowns work across restarts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple


# ---------------
# Data models
# ---------------
@dataclass
class RiskLimits:
    max_leverage: int = 20
    max_stake_usdt: float = 100.0
    max_daily_loss_usdt: float = 300.0
    reopen_cooldown_sec: int = 60          # after any exit
    loss_cooldown_sec: int = 300           # after a loss
    flip_cooldown_sec: int = 120           # after switching side long<->short

    # Optional stake trims based on volatility profile (High/Medium/Low).
    # The orchestrator can pass its vol_profile so we scale stakes gently.
    vol_stake_scale_high: float = 0.8      # 80 percent of max stake in High vol
    vol_stake_scale_medium: float = 1.0    # 100 percent in Medium vol
    vol_stake_scale_low: float = 1.0       # 100 percent in Low vol


@dataclass
class RiskState:
    # Running daily figures and timestamps.
    day_ymd: str = ""                      # e.g., "2025-10-21"
    realized_pnl_usdt: float = 0.0
    last_exit_ts: float = 0.0
    last_loss_ts: float = 0.0
    last_flip_ts: float = 0.0
    last_side: str = ""                    # "long" or "short" when position last existed


# ---------------
# Risk manager
# ---------------
class RiskManager:
    def __init__(self, logger, limits: Optional[RiskLimits] = None, state_path: str = "data/risk_state.json"):
        self.logger = logger
        self.limits = limits or RiskLimits()
        self.state_path = state_path
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state = self._load_state()

    # ---- persistence ----
    def _load_state(self) -> RiskState:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return RiskState(**raw)
        except FileNotFoundError:
            return RiskState(day_ymd=self._today())
        except Exception as e:
            self._safe_log(f"[RISK] failed to load state: {e}")
            return RiskState(day_ymd=self._today())

    def _save_state(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    # ---- basic utilities ----
    def _today(self) -> str:
        # Use UTC date so it is consistent for servers.
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if self.state.day_ymd != today:
            self.state.day_ymd = today
            self.state.realized_pnl_usdt = 0.0
            # Leave timestamps, they still help with rapid re-entries around midnight
            self._save_state()

    def _safe_log(self, msg: str) -> None:
        try:
            if self.logger:
                self.logger.info(msg)
        except Exception:
            pass

    # ---- public API for orchestrator ----
    def check_order(
        self,
        *,
        symbol: str,
        side: str,                     # "long" or "short"
        requested_stake_usdt: float,
        requested_leverage: int,
        vol_profile: str = "Medium",   # "High"|"Medium"|"Low"
        now_ts: Optional[float] = None,
        is_flip: Optional[bool] = None # if orchestrator knows it's flipping
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Decide if a new order is allowed, and what stake to use.

        Returns a tuple:
            (allowed: bool, stake_usdt: float, info: dict)

        info contains reasons, any trims, and remaining cooldowns so the caller
        can print a friendly line and journal it.
        """
        self._roll_day_if_needed()
        now = now_ts if now_ts is not None else time.time()
        reasons = []
        allowed = True

        # 1) Leverage fence
        lev = int(requested_leverage)
        if lev > self.limits.max_leverage:
            reasons.append(f"leverage {lev} exceeds max {self.limits.max_leverage}")
            lev = self.limits.max_leverage

        # 2) Volatility-based stake scale
        v = (vol_profile or "Medium").lower()
        scale = 1.0
        if v == "high":
            scale = self.limits.vol_stake_scale_high
        elif v == "low":
            scale = self.limits.vol_stake_scale_low
        else:
            scale = self.limits.vol_stake_scale_medium

        stake = float(requested_stake_usdt) * float(scale)

        # 3) Max stake fence
        if stake > self.limits.max_stake_usdt:
            reasons.append(f"stake {stake:.2f} > max_stake {self.limits.max_stake_usdt:.2f}, trimming")
            stake = float(self.limits.max_stake_usdt)

        # 4) Daily loss check
        daily_loss = -min(0.0, self.state.realized_pnl_usdt)  # positive number if losing
        remaining_loss_budget = max(0.0, self.limits.max_daily_loss_usdt - daily_loss)
        if remaining_loss_budget <= 0.0:
            reasons.append("max daily loss reached")
            allowed = False

        # 5) Cooldowns: after any exit and after a loss
        remain_exit = self._remaining(self.state.last_exit_ts, self.limits.reopen_cooldown_sec, now)
        if remain_exit > 0:
            reasons.append(f"reopen cooldown {remain_exit}s")
            allowed = False

        remain_loss = self._remaining(self.state.last_loss_ts, self.limits.loss_cooldown_sec, now)
        if remain_loss > 0:
            reasons.append(f"loss cooldown {remain_loss}s")
            allowed = False

        # 6) Flip cooldown
        # If the caller did not specify is_flip, infer it from last_side.
        if is_flip is None:
            is_flip = self.state.last_side and self.state.last_side != side
        remain_flip = self._remaining(self.state.last_flip_ts, self.limits.flip_cooldown_sec, now) if is_flip else 0
        if is_flip and remain_flip > 0:
            reasons.append(f"flip cooldown {remain_flip}s")
            allowed = False

        info = {
            "symbol": symbol,
            "side": side,
            "leverage": lev,
            "stake_usdt": round(stake, 8),
            "requested_stake_usdt": float(requested_stake_usdt),
            "requested_leverage": int(requested_leverage),
            "scaled_by_vol": scale,
            "reasons": reasons,
            "cooldowns": {
                "reopen_sec": max(0, self.limits.reopen_cooldown_sec - int(now - self.state.last_exit_ts)),
                "loss_sec": max(0, self.limits.loss_cooldown_sec - int(now - self.state.last_loss_ts)),
                "flip_sec": max(0, self.limits.flip_cooldown_sec - int(now - self.state.last_flip_ts)) if is_flip else 0,
            },
            "daily_loss_so_far": round(daily_loss, 8),
            "daily_loss_budget_left": round(remaining_loss_budget, 8),
            "vol_profile": vol_profile,
            "is_flip": bool(is_flip),
        }

        # Final allow/deny
        if not allowed:
            self._safe_log(f"[RISK] BLOCK {side} {stake:.2f}USDT lev={lev} | {'; '.join(reasons)}")
            return False, 0.0, info

        # Allow with any trims already applied
        return True, stake, info

    def note_exit(self, *, now_ts: Optional[float] = None) -> None:
        """Call this right after a position is fully closed."""
        now = now_ts if now_ts is not None else time.time()
        self.state.last_exit_ts = now
        self._save_state()

    def note_flip(self, *, now_ts: Optional[float] = None) -> None:
        """Call this if you are closing one side and immediately opening the other."""
        now = now_ts if now_ts is not None else time.time()
        self.state.last_flip_ts = now
        self._save_state()

    def set_last_side(self, side: str) -> None:
        """Remember the last side we held so we can infer flips later."""
        side = (side or "").lower()
        if side in ("long", "short"):
            self.state.last_side = side
            self._save_state()

    def register_fill(self, *, pnl_usdt: float, now_ts: Optional[float] = None) -> None:
        """
        After a trade is closed, tell the risk manager the realised profit or loss.
        This keeps daily loss tracking correct and triggers loss cooldowns if needed.
        """
        self._roll_day_if_needed()
        now = now_ts if now_ts is not None else time.time()
        self.state.realized_pnl_usdt += float(pnl_usdt)

        if pnl_usdt < 0:
            # Start the loss cooldown only when we actually lose money
            self.state.last_loss_ts = now

        self._save_state()

    # ---- helpers ----
    def _remaining(self, start_ts: float, window_sec: int, now: float) -> int:
        if start_ts <= 0 or window_sec <= 0:
            return 0
        remain = window_sec - int(now - start_ts)
        return max(0, remain)


# ---------------
# Minimal self-test
# ---------------
if __name__ == "__main__":
    class _DummyLog:
        def info(self, *a, **k):
            print(*a)

    rm = RiskManager(_DummyLog(), RiskLimits(max_leverage=10, max_stake_usdt=75, max_daily_loss_usdt=50))
    ok, stake, info = rm.check_order(
        symbol="BTC/USDT:USDT",
        side="long",
        requested_stake_usdt=100.0,
        requested_leverage=25,
        vol_profile="High"
    )
    print("allowed:", ok, "stake:", stake, "info:", info)
    if ok:
        # Pretend we took a loss and then check cooldowns kick in.
        rm.register_fill(pnl_usdt=-12.34)
        rm.note_exit()
        ok2, stake2, info2 = rm.check_order(
            symbol="BTC/USDT:USDT",
            side="short",
            requested_stake_usdt=50.0,
            requested_leverage=5,
            vol_profile="Medium",
            is_flip=True
        )
        print("allowed2:", ok2, "stake2:", stake2, "info2:", info2)
