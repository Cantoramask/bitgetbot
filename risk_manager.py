#!/usr/bin/env python3
"""
risk_manager.py
Seatbelt for the Bitget bot.

Plain-English summary:
This file protects you from silly or dangerous trades. Before any order is sent,
the orchestrator asks RiskManager: "Is this safe, and if not, how should I trim it?"
RiskManager checks stake size, daily loss, and cooldowns. It never blocks only because
leverage is high; instead it trims stake and warns you when leverage is aggressive.

Definitions (plain language):
Cooldown means a waiting time after an action before doing the next one.
Daily loss limit means the most money you’re willing to lose in one day.
Flip means closing a long and immediately opening a short, or the reverse.
Reduce-only means an order that can only reduce or close a position, never add to it.
Environment variable means a setting stored outside the code in your .env file.

You can tune defaults with environment variables like:
  RISK_MAX_NOTIONAL_DEFAULT=5000
  RISK_MAX_STAKE_DEFAULT=200
  RISK_LEVERAGE_WARN_X=30
  RISK_MAX_NOTIONAL_SOL=8000
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Any, Tuple

@dataclass
class RiskLimits:
    # No hard max leverage. We remove that concept and scale by leverage instead.
    max_stake_usdt: float = 100.0
    max_daily_loss_usdt: float = 300.0
    reopen_cooldown_sec: int = 60          # after any exit
    loss_cooldown_sec: int = 300           # after a loss
    flip_cooldown_sec: int = 120           # after switching side long<->short
    max_notional_usdt: float = 2_000.0     # cap stake * leverage
    loss_cd_scale_base_lev: int = 5        # scale loss cooldown around this lev

    # Volatility aware stake trims
    vol_stake_scale_high: float = 0.8
    vol_stake_scale_medium: float = 1.0
    vol_stake_scale_low: float = 1.0

@dataclass
class RiskState:
    day_ymd: str
    realized_pnl_usdt: float = 0.0
    last_exit_ts: float = 0.0
    last_loss_ts: float = 0.0
    last_flip_ts: float = 0.0
    last_side: Optional[str] = None  # "long" | "short"

class RiskManager:
    def __init__(self, logger, limits: Optional[RiskLimits] = None, state_path: str = "data/risk_state.json"):
        self.logger = logger
        self.limits = limits or RiskLimits()
        self.state_path = state_path
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state = self._load_state()

        # Per-symbol overrides via env
        #   RISK_MAX_STAKE_DEFAULT, RISK_MAX_NOTIONAL_DEFAULT
        #   RISK_MAX_STAKE_SOL,     RISK_MAX_NOTIONAL_SOL
        self._overrides = self._load_overrides()

        # Leverage warning threshold, not a cap
        self._lev_warn_x = int(os.getenv("RISK_LEVERAGE_WARN_X", "30"))

        # Anti-log-spam
        self._last_block_sig: str = ""
        self._last_block_log_ts: float = 0.0
        self._block_log_every_sec: int = int(os.getenv("RISK_BLOCK_LOG_EVERY_SEC", "5"))

    # ---- persistence ----
    def _load_state(self) -> RiskState:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return RiskState(**raw)
        except FileNotFoundError:
            return RiskState(day_ymd=self._today())
        except Exception:
            return RiskState(day_ymd=self._today())

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f)
        except Exception:
            pass

    # ---- helpers ----
    def _safe_log(self, msg: str) -> None:
        try:
            if self.logger:
                self.logger.info(msg)
        except Exception:
            pass

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if self.state.day_ymd != today:
            self.state.day_ymd = today
            self.state.realized_pnl_usdt = 0.0
            self._save_state()

    def _remaining(self, last_ts: float, cd: int, now: float) -> int:
        if cd <= 0:
            return 0
        if last_ts <= 0:
            return 0
        remain = cd - int(now - last_ts)
        return max(0, remain)

    def _norm_symbol_key(self, symbol: str) -> str:
        s = (symbol or "").upper().replace("/", "").replace(":USDT", "")
        if s.endswith("USDT"):
            return s
        return s + "USDT" if s and not s.endswith("USDT") else s

    def _load_overrides(self) -> Dict[str, Dict[str, float]]:
        ov: Dict[str, Dict[str, float]] = {}

        def _f(name: str) -> Optional[float]:
            v = os.getenv(name)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        d_max_stake = _f("RISK_MAX_STAKE_DEFAULT")
        d_max_notional = _f("RISK_MAX_NOTIONAL_DEFAULT")
        if d_max_stake or d_max_notional:
            ov["__DEFAULT__"] = {}
            if d_max_stake is not None:
                ov["__DEFAULT__"]["max_stake_usdt"] = d_max_stake
            if d_max_notional is not None:
                ov["__DEFAULT__"]["max_notional_usdt"] = d_max_notional

        # Per-coin overrides like RISK_MAX_STAKE_SOL, RISK_MAX_NOTIONAL_SOL
        for k, v in os.environ.items():
            if not (k.startswith("RISK_MAX_STAKE_") or k.startswith("RISK_MAX_NOTIONAL_")):
                continue
            parts = k.split("_")
            coin = parts[-1].upper()
            coin = coin.replace(":", "").replace("/", "")
            coin = coin if coin.endswith("USDT") else coin + "USDT"
            ov.setdefault(coin, {})
            try:
                if k.startswith("RISK_MAX_STAKE_"):
                    ov[coin]["max_stake_usdt"] = float(v)
                elif k.startswith("RISK_MAX_NOTIONAL_"):
                    ov[coin]["max_notional_usdt"] = float(v)
            except Exception:
                pass
        return ov

    def _apply_overrides(self, symbol: str, base: RiskLimits) -> RiskLimits:
        key = self._norm_symbol_key(symbol)
        limits = replace(base)
        dflt = self._overrides.get("__DEFAULT__", {})
        if "max_stake_usdt" in dflt:
            limits.max_stake_usdt = float(dflt["max_stake_usdt"])
        if "max_notional_usdt" in dflt:
            limits.max_notional_usdt = float(dflt["max_notional_usdt"])
        coin = self._overrides.get(key, {})
        if "max_stake_usdt" in coin:
            limits.max_stake_usdt = float(coin["max_stake_usdt"])
        if "max_notional_usdt" in coin:
            limits.max_notional_usdt = float(coin["max_notional_usdt"])
        return limits

    def _lev_stake_scale(self, leverage: int) -> float:
        """
        As leverage increases, reduce stake so notional stays sane.
        We aim roughly stake * leverage <= max_notional.
        """
        try:
            lev = max(1, int(leverage))
        except Exception:
            lev = 1
        # Scale factor reduces stake at high lev, but never kills it.
        # 1x -> 1.0, 5x -> ~1.0, 20x -> 0.5, 40x -> 0.3
        return max(0.2, min(1.0, 5.0 / max(5.0, float(lev))))

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
        is_flip: Optional[bool] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Decide if a new order is allowed, and what stake to use.

        Returns (allowed: bool, stake_usdt: float, info: dict)

        info.reasons holds trims/blocks, info.warnings holds non-blocking warnings.
        """
        self._roll_day_if_needed()
        now = now_ts if now_ts is not None else time.time()
        reasons = []
        warnings = []
        allowed = True

        limits = self._apply_overrides(symbol, self.limits)

        # Leverage warning only. No hard cap. We never block solely due to lev.
        lev = int(requested_leverage)
        if lev >= self._lev_warn_x:
            warnings.append(f"high leverage {lev}x >= warn {self._lev_warn_x}x")

        # Volatility and leverage scaling for stake
        v = (vol_profile or "Medium").lower()
        if v == "high":
            scale_vol = limits.vol_stake_scale_high
        elif v == "low":
            scale_vol = limits.vol_stake_scale_low
        else:
            scale_vol = limits.vol_stake_scale_medium

        scale_lev = self._lev_stake_scale(lev)
        stake = float(requested_stake_usdt) * float(scale_vol) * float(scale_lev)

        # Max stake fence
        if stake > limits.max_stake_usdt:
            reasons.append(f"stake {stake:.2f} > max_stake {limits.max_stake_usdt:.2f}, trimming")
            stake = float(limits.max_stake_usdt)

        # Notional fence: stake * lev
        notional = stake * lev
        if notional > limits.max_notional_usdt:
            reasons.append(f"notional {notional:.2f} > max_notional {limits.max_notional_usdt:.2f}, trimming")
            stake = limits.max_notional_usdt / max(1, lev)
            notional = stake * lev

        # Daily loss check
        daily_loss = -min(0.0, self.state.realized_pnl_usdt)
        remaining_loss_budget = max(0.0, limits.max_daily_loss_usdt - daily_loss)
        if remaining_loss_budget <= 0.0:
            reasons.append("max daily loss reached")
            allowed = False

        # Cooldowns
        remain_exit = self._remaining(self.state.last_exit_ts, limits.reopen_cooldown_sec, now)
        if remain_exit > 0:
            reasons.append(f"reopen cooldown {remain_exit}s")
            allowed = False

        eff_loss_cd = self._scaled_loss_cd(limits.loss_cooldown_sec, lev, limits.loss_cd_scale_base_lev)
        remain_loss = self._remaining(self.state.last_loss_ts, eff_loss_cd, now)
        if remain_loss > 0:
            reasons.append(f"loss cooldown {remain_loss}s")
            allowed = False

        # Flip cooldown
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
            "warnings": warnings,
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
                "max_stake_usdt": limits.max_stake_usdt,
                "max_daily_loss_usdt": limits.max_daily_loss_usdt,
                "max_notional_usdt": limits.max_notional_usdt,
            },
        }

        if not allowed:
            sig = f"{side}|{round(stake,2)}|{'|'.join(reasons)}"
            now_ts = time.time()
            if sig != self._last_block_sig or (now_ts - self._last_block_log_ts) >= self._block_log_every_sec:
                self._safe_log(f"[RISK] BLOCK {side} {stake:.2f}USDT | {'; '.join(reasons)}")
                self._last_block_sig = sig
                self._last_block_log_ts = now_ts
            return False, 0.0, info

        return True, stake, info

    def _scaled_loss_cd(self, base_cd: int, lev: int, base_lev: int) -> int:
        # At higher lev, hold you back a bit longer after a loss
        try:
            lev = max(1, int(lev))
            base_lev = max(1, int(base_lev))
        except Exception:
            return base_cd
        factor = min(2.5, max(0.75, lev / float(base_lev)))
        return int(round(base_cd * factor))

    # ---- notes from orchestrator ----
    def register_fill(self, *, pnl_usdt: float) -> None:
        self.state.realized_pnl_usdt = float(self.state.realized_pnl_usdt) + float(pnl_usdt or 0.0)
        if pnl_usdt < 0:
            self.state.last_loss_ts = time.time()
        self._save_state()

    def note_exit(self, *, now_ts: Optional[float] = None) -> None:
        now = now_ts if now_ts is not None else time.time()
        self.state.last_exit_ts = now
        self._save_state()

    def note_flip(self, *, now_ts: Optional[float] = None) -> None:
        now = now_ts if now_ts is not None else time.time()
        self.state.last_flip_ts = now
        self._save_state()

    def set_last_side(self, side: str) -> None:
        side = (side or "").lower()
        if side in ("long", "short"):
            self.state.last_side = side
            self._save_state()
