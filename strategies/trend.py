#!/usr/bin/env python3
"""
strategies/trend.py
Simple, transparent trend calculator: fast/slow SMA with ATR-like volatility.
Outputs: side in {"long","short","wait"}, reason, trail_pct, caution.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional

class TrendStrategy:
    def __init__(self, *, fast: int = 20, slow: int = 50, atr_len: int = 14,
                 min_trail: float = 0.003, max_trail: float = 0.015):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.atr_len = atr_len
        self.min_trail = float(min_trail)
        self.max_trail = float(max_trail)
        self._prices: Deque[float] = deque(maxlen=max(slow, atr_len) + 2)

    def update_tick(self, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        self._prices.append(float(price))

    def _sma(self, n: int) -> Optional[float]:
        if len(self._prices) < n:
            return None
        total = 0.0
        data = list(self._prices)[-n:]
        for v in data:
            total += v
        return total / n

    def _atr_pct(self) -> Optional[float]:
        # Mean absolute diff of successive closes over atr_len, as a pct of price
        if len(self._prices) < self.atr_len + 1:
            return None
        diffs = []
        p = list(self._prices)[-self.atr_len-1:]
        for i in range(1, len(p)):
            a = p[i - 1]
            b = p[i]
            diffs.append(abs(b - a) / max(1.0, a))
        return (sum(diffs) / len(diffs)) if diffs else 0.0

    def decide(self) -> Dict[str, object]:
        fast = self._sma(self.fast)
        slow = self._sma(self.slow)
        if fast is None or slow is None:
            return {"side": "wait", "reason": "warming_up", "trail_pct": self.min_trail, "caution": None}

        atrp = self._atr_pct() or 0.0
        # Base trail uses 2× ATR proxy, clamped
        trail = max(self.min_trail, min(self.max_trail, 2.0 * atrp))

        # Mild percentage band to avoid whipsaw
        band_pct = 0.0002  # ~2 bps
        if fast > slow * (1.0 + band_pct):
            side = "long"
            reason = "fast_above_slow"
        elif fast < slow * (1.0 - band_pct):
            side = "short"
            reason = "fast_below_slow"
        else:
            side = "wait"
            reason = "near_equilibrium"

        caution = "elevated_volatility" if atrp > 0.01 else None

        return {
            "side": side,
            "reason": reason,
            "trail_pct": round(trail, 6),
            "caution": caution,
        }
