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
from dataclasses import dataclass, asdict, replace
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
    # New leverage aware limits
    max_notional_usdt: float = 2_000.0     # cap stake * leverage
    loss_cd_scale_base_lev: int = 5        # scale loss cooldown around this lev

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

        # Per-symbol overrides loaded from environment once at startup.
        # Examples:
        #   RISK_MAX_LEVERAGE_DEFAULT=30
        #   RISK_MAX_STAKE_DEFAULT=200
        #   RISK_MAX_NOTIONAL_DEFAULT=5000
        #   RISK_MAX_LEVERAGE_SOL=100
        #   RISK_MAX_STAKE_SOL=500
        #   RISK_MAX_NOTIONAL_SOL=15000
        self._overrides = self._load_overrides()

        # Reduce console spam when orders are repeatedly blocked during cooldowns.
        self._last_block_sig: Optional[str] = None
        self._last_block_log_ts: float = 0.0
        self._block_log_every_sec: int = int(os.getenv("RISK_BLOCK_LOG_EVERY_SEC", "5"))  # log at most every 5s

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

    def _norm_symbol_key(self, symbol: str) -> str:
        """Normalise symbols to a simple key like 'SOL' or 'SOLUSDT'.
        Accepts 'SOL/USDT:USDT', 'SOL/USDT', 'SOLUSDT', 'SOL'."""
        s = (symbol or "").upper().strip()
        s = s.replace(":USDT", "")
        s = s.replace("/", "")
        # If it ends with USDT or USD, keep base only for overrides keyed by coin.
        if s.endswith("USDT") and len(s) > 4:
            return s[:-4]  # SOL
        if s.endswith("USD") and len(s) > 3:
            return s[:-3]
        return s  # e.g. SOL

    def _load_overrides(self) -> Dict[str, Dict[str, float]]:
        """Load per-symbol and default overrides from environment variables."""
        ov: Dict[str, Dict[str, float]] = {}
        def _getf(name: str) -> Optional[float]:
            v = os.getenv(name)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        # Defaults
        default_max_lev = _getf("RISK_MAX_LEVERAGE_DEFAULT")
        default_max_stake = _getf("RISK_MAX_STAKE_DEFAULT")
        default_max_notional = _getf("RISK_MAX_NOTIONAL_DEFAULT")
        if default_max_lev or default_max_stake or default_max_notional:
            ov["__DEFAULT__"] = {}
            if default_max_lev:
                ov["__DEFAULT__"]["max_leverage"] = int(default_max_lev)
            if default_max_stake:
                ov["__DEFAULT__"]["max_stake_usdt"] = float(default_max_stake)
            if default_max_notional:
                ov["__DEFAULT__"]["max_notional_usdt"] = float(default_max_notional)

        # Per-coin like RISK_MAX_LEVERAGE_SOL, RISK_MAX_STAKE_SOL, RISK_MAX_NOTIONAL_SOL
        for k, v in os.environ.items():
            if not (k.startswith("RISK_MAX_LEVERAGE_") or k.startswith("RISK_MAX_STAKE_") or k.startswith("RISK_MAX_NOTIONAL_")):
                continue
            parts = k.split("_")
            if len(parts) < 3:
                continue
            kind = "_".join(parts[:3]) if parts[2] == "MAX" else "_".join(parts[:2])
            # normalise coin tail
            coin = "_".join(parts[3:]) if parts[2] == "DEFAULT" else "_".join(parts[2:])
            coin = coin.strip().upper()
            if not coin:
                continue
            ov.setdefault(coin, {})
            if k.startswith("RISK_MAX_LEVERAGE_"):
                try:
                    ov[coin]["max_leverage"] = int(float(v))
                except Exception:
                    pass
            elif k.startswith("RISK_MAX_STAKE_"):
                try:
                    ov[coin]["max_stake_usdt"] = float(v)
                except Exception:
                    pass
            elif k.startswith("RISK_MAX_NOTIONAL_"):
                try:
                    ov[coin]["max_notional_usdt"] = float(v)
                except Exception:
                    pass
        return ov

    def _apply_overrides(self, symbol: str, base: RiskLimits) -> RiskLimits:
        """Return a new RiskLimits with per-symbol or default overrides applied."""
        key = self._norm_symbol_key(symbol)
        limits = replace(base)
        # Default overrides first
        dflt = self._overrides.get("__DEFAULT__", {})
        if "max_leverage" in dflt:
            limits.max_leverage = int(dflt["max_leverage"])
        if "max_stake_usdt" in dflt:
            limits.max_stake_usdt = float(dflt["max_stake_usdt"])
        if "max_notional_usdt" in dflt:
            limits.max_notional_usdt = float(dflt["max_notional_usdt"])
        # Per-coin overrides next
        ov = self._overrides.get(key, {})
        if "max_leverage" in ov:
            limits.max_leverage = int(ov["max_leverage"])
        if "max_stake_usdt" in ov:
            limits.max_stake_usdt = float(ov["max_stake_usdt"])
        if "max_notional_usdt" in ov:
            limits.max_notional_usdt = float(ov["max_notional_usdt"])
        return limits

    # ---- leverage aware helpers ----
    def _lev_stake_scale(self, lev: int) -> float:
        # halve stake at lev=4 to 5 and continue to decrease sublinearly
        l = max(1, int(lev))
        return 1.0 / (l ** 0.5)

    def _scaled_loss_cd(self, base_cd: int, lev: int, base_lev: int) -> int:
        if base_cd <= 0:
            return 0
        l = max(1, int(lev))
        b = max(1, int(base_lev))
        factor = max(1.0, (l / b) ** 0.5)
        return int(round(base_cd * factor))

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
        """Decide if a new order is allowed, and what stake to use.

        Returns a tuple:
            (allowed: bool, stake_usdt: float, info: dict)

        info contains reasons, any trims, and remaining cooldowns so the caller
        can print a friendly line and journal it.
        """
        self._roll_day_if_needed()
        now = now_ts if now_ts is not None else time.time()
        reasons = []
        allowed = True

        # Apply per-symbol overrides to the base limits.
        limits = self._apply_overrides(symbol, self.limits)

        # 1) Leverage fence
        lev = int(requested_leverage)
        if lev > limits.max_leverage:
            reasons.append(f"leverage {lev} exceeds max {limits.max_leverage}")
            lev = limits.max_leverage

        # 2) Volatility-based stake scale and leverage-based scale
        v = (vol_profile or "Medium").lower()
        scale_vol = 1.0
        if v == "high":
            scale_vol = limits.vol_stake_scale_high
        elif v == "low":
            scale_vol = limits.vol_stake_scale_low
        else:
            scale_vol = limits.vol_stake_scale_medium

        scale_lev = self._lev_stake_scale(lev)
        stake = float(requested_stake_usdt) * float(scale_vol) * float(scale_lev)

        # 3) Max stake fence
        if stake > limits.max_stake_usdt:
            reasons.append(f"stake {stake:.2f} > max_stake {limits.max_stake_usdt:.2f}, trimming")
            stake = float(limits.max_stake_usdt)

        # 4) Notional fence
        notional = stake * lev
        if notional > limits.max_notional_usdt:
            reasons.append(f"notional {notional:.2f} > max_notional {limits.max_notional_usdt:.2f}, trimming")
            stake = limits.max_notional_usdt / max(1, lev)
            notional = stake * lev

        # 5) Daily loss check
        daily_loss = -min(0.0, self.state.realized_pnl_usdt)  # positive number if losing
        remaining_loss_budget = max(0.0, limits.max_daily_loss_usdt - daily_loss)
        if remaining_loss_budget <= 0.0:
            reasons.append("max daily loss reached")
            allowed = False

        # 6) Cooldowns: after any exit and after a loss with leverage aware scaling
        remain_exit = self._remaining(self.state.last_exit_ts, limits.reopen_cooldown_sec, now)
        if remain_exit > 0:
            reasons.append(f"reopen cooldown {remain_exit}s")
            allowed = False

        eff_loss_cd = self._scaled_loss_cd(limits.loss_cooldown_sec, lev, limits.loss_cd_scale_base_lev)
        remain_loss = self._remaining(self.state.last_loss_ts, eff_loss_cd, now)
        if remain_loss > 0:
            reasons.append(f"loss cooldown {remain_loss}s")
            allowed = False

        # 7) Flip cooldown
        # If the caller did not specify is_flip, infer it from last_side.
        if is_flip is None:
            is_flip = self.state.last_side and self.state.last_side != side
        remain_flip = self._remaining(self.state.last_flip_ts, limits.flip_cooldown_sec, now) if is_flip else 0
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
            "scaled_by_vol": scale_vol,
            "scaled_by_leverage": scale_lev,
            "effective_notional_usdt": round(notional, 8),
            "reasons": reasons,
            "cooldowns": {
                "reopen_sec": max(0, limits.reopen_cooldown_sec - int(now - self.state.last_exit_ts)),
                "loss_sec": max(0, eff_loss_cd - int(now - self.state.last_loss_ts)),
                "flip_sec": max(0, limits.flip_cooldown_sec - int(now - self.state.last_flip_ts)) if is_flip else 0,
            },
            "daily_loss_so_far": round(daily_loss, 8),
            "daily_loss_budget_left": round(remaining_loss_budget, 8),
            "vol_profile": vol_profile,
            "is_flip": bool(is_flip),
            "applied_limits": {
                "max_leverage": limits.max_leverage,
                "max_stake_usdt": limits.max_stake_usdt,
                "max_notional_usdt": limits.max_notional_usdt,
                "max_daily_loss_usdt": limits.max_daily_loss_usdt,
            },
        }

        # Final allow/deny
        if not allowed:
            # Build a signature of the block reasons to avoid spamming logs.
            sig = f"{side}|{lev}|{round(stake,2)}|{'|'.join(reasons)}"
            now_ts = time.time()
            if sig != self._last_block_sig or (now_ts - self._last_block_log_ts) >= self._block_log_every_sec:
                self._safe_log(f"[RISK] BLOCK {side} {stake:.2f}USDT lev={lev} | {'; '.join(reasons)}")
                self._last_block_sig = sig
                self._last_block_log_ts = now_ts
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

    # Example: allow bigger leverage and stake by default and specifically for SOL.
    os.environ.setdefault("RISK_MAX_LEVERAGE_DEFAULT", "30")
    os.environ.setdefault("RISK_MAX_STAKE_DEFAULT", "200")
    os.environ.setdefault("RISK_MAX_LEVERAGE_SOL", "100")
    os.environ.setdefault("RISK_MAX_STAKE_SOL", "500")

    rm = RiskManager(_DummyLog(), RiskLimits(max_leverage=20, max_stake_usdt=100, max_daily_loss_usdt=300))
    ok, stake, info = rm.check_order(
        symbol="SOL/USDT:USDT",
        side="long",
        requested_stake_usdt=120.0,
        requested_leverage=30,
        vol_profile="Medium"
    )
    print("allowed:", ok, "stake:", stake, "info:", {k: info[k] for k in ('leverage','stake_usdt','reasons','applied_limits')})
    if ok:
        rm.register_fill(pnl_usdt=-12.34)
        rm.note_exit()
        ok2, stake2, info2 = rm.check_order(
            symbol="SOL/USDT:USDT",
            side="short",
            requested_stake_usdt=50.0,
            requested_leverage=5,
            vol_profile="Medium",
            is_flip=True
        )
        print("allowed2:", ok2, "stake2:", stake2, "info2:", info2)
