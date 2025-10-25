#!/usr/bin/env python3
"""
strategies/trend.py
Fast/slow SMA trend with volatility-adaptive deadband and leverage-aware trails.
Outputs: side in {"long","short","wait"}, reason, trail_pct, caution.
"""

from __future__ import annotations

import math
import os
import statistics
import collections
from typing import Deque, Dict, Optional, List


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TrendStrategy:
    def __init__(
        self,
        *,
        fast: int = 9,
        slow: int = 21,
        atr_len: int = 14,
        min_trail: float = 0.002,   # 0.20%
        max_trail: float = 0.010,   # 1.00%
    ):
        """
        fast/slow are SMA lengths (in ticks). Smaller = more trades.
        Defaults (9,21) are active on 1s ticks.
        """
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = int(fast)
        self.slow = int(slow)
        self.atr_len = max(5, int(atr_len))
        self.min_trail = float(min_trail)
        self.max_trail = float(max_trail)

        # Leverage hint (optional). If not set via set_leverage, falls back to env.
        try:
            self._lev = max(1, int(os.getenv("LEVERAGE", "5")))
        except Exception:
            self._lev = 5

        # internal state
        self._prices: Deque[float] = collections.deque(maxlen=max(self.slow, self.atr_len) + 200)
        self._vol_slope_status: Optional[str] = None  # "up" | "down" | "flat"
        self._last_side: str = "wait"

        # Optional deadband multiplier via ENV (e.g. 0.5 tighter, 2.0 looser)
        try:
            self._band_mult = float(os.getenv("TREND_BAND_MULT", "1.0"))
        except Exception:
            self._band_mult = 1.0

    # ---- config helpers -----------------------------------------------------

    def set_windows(self, fast: int, slow: int) -> None:
        if fast < slow and fast > 0:
            self.fast = int(fast)
            self.slow = int(slow)
            self._prices = collections.deque(self._prices, maxlen=max(self.slow, self.atr_len) + 200)

    def set_atr_len(self, n: int) -> None:
        self.atr_len = max(5, int(n))
        self._prices = collections.deque(self._prices, maxlen=max(self.slow, self.atr_len) + 200)

    def set_leverage(self, lev: int) -> None:
        try:
            self._lev = max(1, int(lev))
        except Exception:
            pass

    # ---- streaming updates --------------------------------------------------

    def update_tick(self, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        self._prices.append(float(price))

    # ---- indicators ---------------------------------------------------------

    def _sma(self, n: int) -> Optional[float]:
        if len(self._prices) < n:
            return None
        it = list(self._prices)[-n:]
        return sum(it) / n

    def _atr_pct(self) -> Optional[float]:
        """ATR proxy: mean absolute close-to-close change (percentage)."""
        if len(self._prices) < self.atr_len + 1:
            return None
        diffs = []
        p = list(self._prices)[-self.atr_len - 1 :]
        for i in range(1, len(p)):
            a = p[i - 1]
            b = p[i]
            diffs.append(abs(b - a) / max(1.0, a))
        return (sum(diffs) / len(diffs)) if diffs else 0.0

    def _pct_changes(self) -> List[float]:
        ps = list(self._prices)
        out: List[float] = []
        for i in range(1, len(ps)):
            a = ps[i - 1]
            b = ps[i]
            if a > 0:
                out.append((b - a) / a)
        return out

    def _std_pct(self, n: int) -> Optional[float]:
        ch = self._pct_changes()
        if len(ch) < n:
            return None
        seg = ch[-n:]
        try:
            return float(statistics.pstdev(seg))
        except Exception:
            return None

    def _vol_slope(self) -> Optional[str]:
        """
        Compare long window std (60) to short window std (10).
        If long > short by a small band, volatility is rising ("up").
        If long < short by a small band, volatility is easing ("down").
        Else "flat".
        """
        long_std = self._std_pct(60)
        short_std = self._std_pct(10)
        if long_std is None or short_std is None:
            return None
        band = max(1e-6, short_std * 0.05)  # 5% band of the short std
        if long_std > short_std + band:
            return "up"
        if long_std < short_std - band:
            return "down"
        return "flat"

    @property
    def vol_slope_status(self) -> Optional[str]:
        return self._vol_slope_status

    # ---- decision -----------------------------------------------------------

    def decide(self) -> Dict[str, object]:
        fast = self._sma(self.fast)
        slow = self._sma(self.slow)
        if fast is None or slow is None:
            self._vol_slope_status = None
            return {
                "side": "wait",
                "reason": "warming_up",
                "trail_pct": self.min_trail,
                "caution": None,
            }

        # --- Volatility + leverage aware trail
        atrp = self._atr_pct() or 0.0  # e.g. 8e-06 ~ 0.0008%
        # Base trail ~ 2×ATR (percentage), clamped
        base_trail = _clamp(2.0 * atrp, self.min_trail, self.max_trail)
        # Tighten with leverage (square-root law)
        lev_tight = base_trail / (self._lev ** 0.5)
        trail = _clamp(lev_tight, self.min_trail, self.max_trail)

        # Volatility regime tweak
        slope = self._vol_slope()
        self._vol_slope_status = slope
        caution: Optional[str] = None
        if slope == "up":
            trail = _clamp(trail * 0.9, self.min_trail, self.max_trail)   # tighten a touch
            caution = "elevated_volatility"
        elif slope == "down":
            trail = _clamp(trail * 1.05, self.min_trail, self.max_trail)  # allow a bit wider

        # --- Adaptive deadband (ATR-scaled) for crossover
        # Make it easy to leave the equilibrium zone so entries actually occur.
        # Band is half an ATR (in pct) with sensible caps; also allow ENV multiplier.
        band_pct = _clamp(0.5 * atrp * self._band_mult, 0.00003, 0.0008)  # 0.003% .. 0.08%
        # Momentum escape: reduce band slightly if volatility is falling
        if slope == "down":
            band_pct *= 0.8

        diff_pct = (fast - slow) / slow if slow != 0 else 0.0

        # Hysteresis for stability: once a side is chosen, keep it until diff re-enters
        # a slightly smaller inner band to avoid immediate flip-flop.
        inner_band = band_pct * 0.7

        side = "wait"
        reason = "near_equilibrium"

        if diff_pct > band_pct:
            side, reason = "long", "fast_above_slow"
        elif diff_pct < -band_pct:
            side, reason = "short", "fast_below_slow"
        else:
            # If previously trending and still outside the inner band, keep the bias.
            if self._last_side == "long" and diff_pct > inner_band:
                side, reason = "long", "bias_hold_long"
            elif self._last_side == "short" and diff_pct < -inner_band:
                side, reason = "short", "bias_hold_short"
            else:
                side, reason = "wait", "near_equilibrium"

        self._last_side = side

        return {
            "side": side,
            "reason": reason,
            "trail_pct": round(trail, 6),
            "caution": caution,
        }