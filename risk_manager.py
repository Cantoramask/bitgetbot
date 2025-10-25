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
  RISK_MIN_STAKE_DEFAULT=5
  RISK_LEVERAGE_WARN_X=30
  RISK_LEV_KNEE=5
  RISK_MAX_NOTIONAL_SOL=8000
"""

from __future__ import annotations

import json
import os
import time
import random
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Any, Tuple


@dataclass
class RiskLimits:
    # No hard max leverage. We remove that concept and scale by leverage instead.
    max_stake_usdt: float = 100.0
    min_stake_usdt: float = 0.0
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
        #   RISK_MAX_STAKE_DEFAULT, RISK_MAX_NOTIONAL_DEFAULT, RISK_MIN_STAKE_DEFAULT
        #   RISK_MAX_STAKE_SOL,     RISK_MAX_NOTIONAL_SOL
        self._overrides = self._load_overrides()

        # Leverage warning threshold, not a cap
        self._lev_warn_x = int(os.getenv("RISK_LEVERAGE_WARN_X", "30"))

        # Tunable leverage knee for stake scaling
        self._lev_knee = max(1.0, float(os.getenv("RISK_LEV_KNEE", "5")))

        # Flip hard-block behaviour within fee window
        self._flip_hardblock_within_fees = os.getenv("RISK_FLIP_HARDBLOCK_WITHIN_FEES", "1").strip() not in ("0", "false", "False")
        self._flip_block_window_sec = int(os.getenv("RISK_FLIP_BLOCK_WINDOW_SEC", "20"))  # very short window
        self._fee_buffer_mult = float(os.getenv("RISK_FEE_BUFFER_MULT", "1.0"))           # multiply tx_cost_bps

        # Dynamic cooldown scaling from realised vol/ATR
        self._atr_base_frac = float(os.getenv("RISK_ATR_BASE_FRAC", "0.003"))             # 0.3% baseline
        self._cd_scale_min = float(os.getenv("RISK_CD_SCALE_MIN", "0.7"))
        self._cd_scale_max = float(os.getenv("RISK_CD_SCALE_MAX", "1.8"))

        # Context staleness handling
        self._stale_stake_scale = float(os.getenv("RISK_STALE_STAKE_SCALE", "0.8"))       # cut stake when ctx stale
        self._stale_cd_mult = float(os.getenv("RISK_STALE_CD_MULT", "1.3"))               # lengthen cooldowns

        # Debounced state writes with tiny jitter to avoid thundering herds
        self._save_debounce_sec = float(os.getenv("RISK_SAVE_DEBOUNCE_SEC", "0.5"))
        self._save_jitter_ms = max(0.0, float(os.getenv("RISK_SAVE_JITTER_MS", "15")))
        self._last_save_ts: float = 0.0

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

    def _save_state(self, *, force: bool = False) -> None:
        try:
            now = time.time()
            if not force and (now - self._last_save_ts) < self._save_debounce_sec:
                return
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f)
            # push next-allowed save slightly into the future to desynchronise bursts
            self._last_save_ts = now + (random.uniform(0.0, self._save_jitter_ms) / 1000.0)
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
            self._save_state(force=True)

    def _remaining(self, last_ts: float, cd: int, now: float) -> int:
        if cd <= 0 or last_ts <= 0:
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
        d_min_stake = _f("RISK_MIN_STAKE_DEFAULT")
        if any(v is not None for v in (d_max_stake, d_max_notional, d_min_stake)):
            ov["__DEFAULT__"] = {}
            if d_max_stake is not None:
                ov["__DEFAULT__"]["max_stake_usdt"] = d_max_stake
            if d_max_notional is not None:
                ov["__DEFAULT__"]["max_notional_usdt"] = d_max_notional
            if d_min_stake is not None:
                ov["__DEFAULT__"]["min_stake_usdt"] = d_min_stake

        # Per-coin overrides like RISK_MAX_STAKE_SOL, RISK_MAX_NOTIONAL_SOL
        for k, v in os.environ.items():
            if not (k.startswith("RISK_MAX_STAKE_") or k.startswith("RISK_MAX_NOTIONAL_") or k.startswith("RISK_MIN_STAKE_")):
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
                elif k.startswith("RISK_MIN_STAKE_"):
                    ov[coin]["min_stake_usdt"] = float(v)
            except Exception:
                pass
        return ov

    def _apply_overrides(self, symbol: str, base: RiskLimits) -> RiskLimits:
        key = self._norm_symbol_key(symbol)
        limits = replace(base)
        dflt = self._overrides.get("__DEFAULT__", {})
        if "max_stake_usdt" in dflt:
            limits.max_stake_usdt = float(dflt["max_stake_usdt"])
        if "min_stake_usdt" in dflt:
            limits.min_stake_usdt = float(dflt["min_stake_usdt"])
        if "max_notional_usdt" in dflt:
            limits.max_notional_usdt = float(dflt["max_notional_usdt"])
        coin = self._overrides.get(key, {})
        if "max_stake_usdt" in coin:
            limits.max_stake_usdt = float(coin["max_stake_usdt"])
        if "min_stake_usdt" in coin:
            limits.min_stake_usdt = float(coin["min_stake_usdt"])
        if "max_notional_usdt" in coin:
            limits.max_notional_usdt = float(coin["max_notional_usdt"])
        return limits

    def _lev_stake_scale(self, leverage: int) -> float:
        """
        As leverage increases, reduce stake so notional stays sane.
        Tunable knee: scale = min(1.0, knee / lev), clamped to [0.2, 1.0]
        """
        try:
            lev = max(1.0, float(leverage))
        except Exception:
            lev = 1.0
        scale = min(1.0, self._lev_knee / lev)
        return max(0.2, float(scale))

    def _dyn_cd_scale_from_atr(self, atr_frac: Optional[float], realized_vol_frac: Optional[float]) -> float:
        """
        Convert ATR% or realised vol% to a bounded cooldown multiplier.
        Baseline at self._atr_base_frac. Above baseline -> longer cooldowns. Below -> shorter.
        """
        v = None
        try:
            if atr_frac is not None:
                v = float(atr_frac)
            elif realized_vol_frac is not None:
                v = float(realized_vol_frac)
        except Exception:
            v = None
        if not v or self._atr_base_frac <= 0:
            return 1.0
        ratio = max(0.0, v / self._atr_base_frac)
        return max(self._cd_scale_min, min(self._cd_scale_max, ratio))

    # ---- public API for orchestrator ----
    def check_order(
        self,
        *,
        symbol: str,
        side: str,                          # "long" or "short"
        requested_stake_usdt: float,
        requested_leverage: int,
        vol_profile: str = "Medium",        # "High"|"Medium"|"Low"
        now_ts: Optional[float] = None,
        is_flip: Optional[bool] = None,
        confidence: Optional[float] = None, # advisor confidence 0..1 for audit
        atr_frac: Optional[float] = None,   # feeder ATR as fraction of price (e.g., 0.004 = 0.4%)
        realized_vol_frac: Optional[float] = None,  # alt to ATR
        tx_cost_bps: Optional[float] = None,        # total round-trip cost in bps
        move_from_entry_bps: Optional[float] = None,# signed bps from entry (bot's current pos)
        is_ctx_stale: Optional[bool] = None         # feeder context staleness flag
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

        # First, deterministic base sizing:
        lev_scale = self._lev_stake_scale(lev)
        cap_stake_by_notional = limits.max_notional_usdt / max(1, lev)
        base_stake = float(requested_stake_usdt) * lev_scale
        # floor before other trims so microscopic stakes aren't produced
        stake = max(float(limits.min_stake_usdt), base_stake)
        # hard caps
        stake = min(stake, float(limits.max_stake_usdt), float(cap_stake_by_notional))

        # Then apply volatility and confidence trims
        v = (vol_profile or "Medium").lower()
        if v == "high":
            scale_vol = limits.vol_stake_scale_high
        elif v == "low":
            scale_vol = limits.vol_stake_scale_low
        else:
            scale_vol = limits.vol_stake_scale_medium

        try:
            conf = float(confidence) if confidence is not None else None
            conf_effect = max(0.5, min(1.0, conf)) if conf is not None else 1.0
        except Exception:
            conf = None
            conf_effect = 1.0

        stake *= float(scale_vol) * float(conf_effect)

        # Context staleness trims (safer when data is old)
        if is_ctx_stale:
            stake *= self._stale_stake_scale

        # Re-impose caps after trims and floors
        stake = max(float(limits.min_stake_usdt), stake)
        stake = min(stake, float(limits.max_stake_usdt), float(cap_stake_by_notional))

        # Effective notional after final stake
        notional = stake * lev

        # Daily loss check
        daily_loss = -min(0.0, self.state.realized_pnl_usdt)
        remaining_loss_budget = max(0.0, limits.max_daily_loss_usdt - daily_loss)
        if remaining_loss_budget <= 0.0:
            reasons.append("max daily loss reached")
            allowed = False

        # Dynamic cooldown scaling from ATR/realised vol (+ staleness)
        cd_dyn = self._dyn_cd_scale_from_atr(atr_frac, realized_vol_frac)
        if is_ctx_stale:
            cd_dyn *= self._stale_cd_mult

        # Reopen cooldown: base by vol_profile, then dynamic ATR/stale scale
        if v == "low":
            reopen_factor = 0.5
        elif v == "high":
            reopen_factor = 1.5
        else:
            reopen_factor = 1.0
        eff_reopen_cd = int(round(limits.reopen_cooldown_sec * reopen_factor * cd_dyn))

        remain_exit = self._remaining(self.state.last_exit_ts, eff_reopen_cd, now)
        if remain_exit > 0:
            reasons.append(f"reopen cooldown {remain_exit}s")
            allowed = False

        # Loss cooldown scaled by leverage and dynamic ATR/stale
        eff_loss_cd_lev = self._scaled_loss_cd(limits.loss_cooldown_sec, lev, limits.loss_cd_scale_base_lev)
        eff_loss_cd = int(round(eff_loss_cd_lev * cd_dyn))
        remain_loss = self._remaining(self.state.last_loss_ts, eff_loss_cd, now)
        if remain_loss > 0:
            reasons.append(f"loss cooldown {remain_loss}s")
            allowed = False

        # Flip cooldown: advisory by default, but optional hard-block inside fees for a short window
        if is_flip is None:
            is_flip = self.state.last_side and self.state.last_side != side
        remain_flip = self._remaining(self.state.last_flip_ts, limits.flip_cooldown_sec, now) if is_flip else 0
        if is_flip and remain_flip > 0:
            if self._flip_hardblock_within_fees:
                try:
                    cost_bps = float(tx_cost_bps) if tx_cost_bps is not None else None
                    move_bps = abs(float(move_from_entry_bps)) if move_from_entry_bps is not None else None
                except Exception:
                    cost_bps, move_bps = None, None
                within_window = (now - self.state.last_flip_ts) <= self._flip_block_window_sec
                if cost_bps is not None and move_bps is not None and within_window and move_bps <= cost_bps * self._fee_buffer_mult:
                    reasons.append(f"flip hard-block within fees for {int(self._flip_block_window_sec - (now - self.state.last_flip_ts))}s")
                    allowed = False
                else:
                    warnings.append(f"flip cooldown {remain_flip}s (allowed during active flip)")
            else:
                warnings.append(f"flip cooldown {remain_flip}s (allowed during active flip)")

        info = {
            "symbol": symbol,
            "side": side,
            "leverage": lev,
            "effective_stake_usdt": round(stake, 8),
            "stake_usdt": round(stake, 8),
            "requested_stake_usdt": float(requested_stake_usdt),
            "requested_leverage": int(requested_leverage),
            "scaled_by_vol": scale_vol,
            "scaled_by_leverage": lev_scale,
            "scaled_by_confidence": None if confidence is None else conf_effect,
            "requested_confidence": conf,
            "effective_notional_usdt": round(notional, 8),
            "reasons": reasons,
            "warnings": warnings,
            "cooldowns": {
                "reopen_sec": max(0, eff_reopen_cd - int(now - self.state.last_exit_ts)),
                "loss_sec": max(0, eff_loss_cd - int(now - self.state.last_loss_ts)),
                "flip_sec": max(0, limits.flip_cooldown_sec - int(now - self.state.last_flip_ts)) if is_flip else 0,
                "tx_cost_bps": tx_cost_bps,
                "move_from_entry_bps": move_from_entry_bps,
                "flip_block_window_sec": self._flip_block_window_sec,
                "fee_buffer_mult": self._fee_buffer_mult,
            },
            "reopen_cd_scaled": eff_reopen_cd,
            "loss_cd_scaled": eff_loss_cd,
            "daily_loss_so_far": round(daily_loss, 8),
            "daily_loss_budget_left": round(remaining_loss_budget, 8),
            "vol_profile": vol_profile,
            "atr_frac": atr_frac,
            "realized_vol_frac": realized_vol_frac,
            "cd_dynamic_scale": cd_dyn,
            "is_ctx_stale": bool(is_ctx_stale) if is_ctx_stale is not None else None,
            "tx_cost_bps": tx_cost_bps,
            "move_from_entry_bps": move_from_entry_bps,
            "is_flip": bool(is_flip),
            "applied_limits": {
                "max_stake_usdt": limits.max_stake_usdt,
                "min_stake_usdt": limits.min_stake_usdt,
                "max_daily_loss_usdt": limits.max_daily_loss_usdt,
                "max_notional_usdt": limits.max_notional_usdt,
            },
        }

        if not allowed:
            sig = f"{side}|{round(stake,2)}|{'|'.join(reasons)}"
            now_ts2 = time.time()
            if sig != self._last_block_sig or (now_ts2 - self._last_block_log_ts) >= self._block_log_every_sec:
                self._safe_log(f"[RISK] BLOCK {side} {stake:.2f}USDT | {'; '.join(reasons)}")
                self._last_block_sig = sig
                self._last_block_log_ts = now_ts2
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

    def note_flip(self, *, now_ts: Optional[float] = None, side: Optional[str] = None) -> None:
        now = now_ts if now_ts is not None else time.time()
        self.state.last_flip_ts = now
        if side:
            s = (side or "").lower()
            if s in ("long", "short"):
                self.state.last_side = s
        self._save_state()

    def set_last_side(self, side: str) -> None:
        side = (side or "").lower()
        if side in ("long", "short"):
            self.state.last_side = side
            self._save_state()
