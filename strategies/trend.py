#!/usr/bin/env python3
"""
strategies/trend.py
Simple, transparent trend calculator: fast/slow SMA with ATR-like volatility.
Outputs: side in {"long","short","wait"}, reason, trail_pct, caution.
"""

from __future__ import annotations

import math
import statistics
import collections
from typing import Deque, Dict, Optional, List

class TrendStrategy:
    def __init__(self, *, fast:int=9, slow:int=21, atr_len:int=14,
                 min_trail:float=0.003, max_trail:float=0.015):
        """
        fast/slow are SMA lengths (in ticks). Smaller = more trades.
        Defaults (9,21) are more active than (20,50) on 1-second ticks.
        """
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.atr_len = atr_len
        self.min_trail = float(min_trail)
        self.max_trail = float(max_trail)
        self._lev = 1
        self._prices: Deque[float] = collections.deque(maxlen=max(slow, self.atr_len) + 120)  # extra room
        self._vol_slope_status: Optional[str] = None  # "up" | "down" | "flat"

    def set_windows(self, fast:int, slow:int) -> None:
        if fast < slow and fast > 0:
            self.fast = int(fast)
            self.slow = int(slow)
            self._prices = collections.deque(self._prices, maxlen=max(self.slow, self.atr_len) + 120)

    def set_atr_len(self, n:int) -> None:
        self.atr_len = max(5, int(n))
        self._prices = collections.deque(self._prices, maxlen=max(self.slow, self.atr_len) + 120)

    def update_tick(self, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        self._prices.append(float(price))

    def _sma(self, n:int) -> Optional[float]:
        if len(self._prices) < n:
            return None
        s = 0.0
        it = list(self._prices)[-n:]
        for v in it:
            s += v
        return s / n

    def _atr_pct(self) -> Optional[float]:
        # Tick-range proxy: mean absolute diff of successive closes over atr_len
        if len(self._prices) < self.atr_len + 1:
            return None
        diffs = []
        p = list(self._prices)[-self.atr_len-1:]
        for i in range(1, len(p)):
            a = p[i-1]; b = p[i]
            diffs.append(abs(b - a) / max(1.0, a))
        return sum(diffs)/len(diffs) if diffs else 0.0

    def _pct_changes(self) -> List[float]:
        ps = list(self._prices)
        out: List[float] = []
        for i in range(1, len(ps)):
            a = ps[i-1]; b = ps[i]
            if a > 0:
                out.append((b - a) / a)
        return out

    def _std_pct(self, n:int) -> Optional[float]:
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

    def decide(self) -> Dict[str, object]:
        fast = self._sma(self.fast)
        slow = self._sma(self.slow)
        if fast is None or slow is None:
            self._vol_slope_status = None
            return {"side": "wait", "reason": "warming_up", "trail_pct": self.min_trail, "caution": None}

        atrp = self._atr_pct() or 0.0
        # Base trail uses 2× ATR proxy, clamped, then tighten by leverage factor
        base_trail = max(self.min_trail, min(self.max_trail, 2.0 * atrp))
        lev_tight = base_trail / (self._lev ** 0.5)
        trail = max(self.min_trail, min(self.max_trail, lev_tight))

        # Volatility slope tweak (predictive modelling lite)
        slope = self._vol_slope()
        self._vol_slope_status = slope
        if slope == "up":
            trail = max(self.min_trail, min(self.max_trail, trail * 0.9))   # tighten a bit
            caution = "elevated_volatility"
        elif slope == "down":
            trail = max(self.min_trail, min(self.max_trail, trail * 1.05))  # allow a touch wider
            caution = None
        else:
            caution = None

        side = "wait"
        reason = "flat"

        # Simple crossover with mild percentage band (≈2 bps) to avoid whipsaw
        band_pct = 0.0002
        if fast > slow * (1.0 + band_pct):
            side = "long"; reason = "fast_above_slow"
        elif fast < slow * (1.0 - band_pct):
            side = "short"; reason = "fast_below_slow"
        else:
            side = "wait"; reason = "near_equilibrium"

        return {
            "side": side,
            "reason": reason,
            "trail_pct": round(trail, 6),
            "caution": caution
        }