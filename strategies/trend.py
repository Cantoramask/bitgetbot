#!/usr/bin/env python3
"""
strategies/trend.py
Simple, transparent trend calculator: fast/slow SMA with ATR-like volatility.
Outputs: side in {"long","short","wait"}, reason, trail_pct, caution.
"""

from __future__ import annotations

import math
import collections
from typing import Deque, Dict, Optional

class TrendStrategy:
    def __init__(self, *, fast:int=20, slow:int=50, atr_len:int=14,
                 min_trail:float=0.003, max_trail:float=0.015):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = int(fast)
        self.slow = int(slow)
        self.atr_len = int(atr_len)
        self.min_trail = float(min_trail)
        self.max_trail = float(max_trail)

        # allow headroom for adaptive window growth
        headroom = max(int(self.slow * 2), self.atr_len) + 5
        self._prices: Deque[float] = collections.deque(maxlen=headroom)

    # ---------- internal helpers ----------
    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def _sma(self, n:int) -> Optional[float]:
        if n <= 0 or len(self._prices) < n:
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

    # ---------- public API ----------
    def update_tick(self, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        self._prices.append(float(price))

    def decide(self) -> Dict[str, object]:
        # 1) Volatility snapshot
        atrp = self._atr_pct()
        if atrp is None:
            # warming up
            return {"side": "wait", "reason": "warming_up", "trail_pct": self.min_trail, "caution": None}

        # 2) Map ATR% into a 0..1 scale between "calm" and "wild"
        #    tune these anchors to your market if you like
        calm, wild = 0.003, 0.02  # ~0.3% .. 2.0% avg tick move
        k = self._clamp((atrp - calm) / (wild - calm), 0.0, 1.0)

        # 3) Adaptive lookbacks: more volatile -> longer windows (smoother),
        #    calmer -> shorter windows (more responsive).
        #    Keep within a safe range so we never exceed buffer length.
        fast_min = max(2, int(round(self.fast * 0.6)))
        fast_max = int(round(self.fast * 1.4))
        slow_min = max(fast_min + 1, int(round(self.slow * 0.6)))
        slow_max = int(round(self.slow * 1.4))

        # scale fast a bit less than slow so ordering stays separated
        eff_fast = int(round(self.fast * (0.8 + 0.6 * k)))
        eff_slow = int(round(self.slow * (0.7 + 0.8 * k)))

        eff_fast = int(self._clamp(eff_fast, fast_min, fast_max))
        eff_slow = int(self._clamp(eff_slow, slow_min, slow_max))

        if eff_fast >= eff_slow:
            # enforce separation
            eff_fast = max(2, eff_slow - 1)

        # Not enough bars yet for adaptive windows? wait.
        if len(self._prices) < eff_slow:
            return {"side": "wait", "reason": "warming_up", "trail_pct": self.min_trail, "caution": None}

        # 4) Compute MAs
        fast = self._sma(eff_fast)
        slow = self._sma(eff_slow)
        if fast is None or slow is None:
            return {"side": "wait", "reason": "warming_up", "trail_pct": self.min_trail, "caution": None}

        # 5) Adaptive trail: base on ATR proxy, clamped to min/max
        trail = max(self.min_trail, min(self.max_trail, 2.0 * atrp))

        # 6) Adaptive band: widen in high vol, narrow in low vol
        #    baseline band ≈ 2 bps of price; scale up to ~15 bps in wild moves
        base_band = 0.0002
        band = base_band * (1.0 + 7.0 * k)            # ~0.02% .. ~0.16%

        # 7) Signal
        side = "wait"
        reason = "near_equilibrium"
        if fast > slow * (1.0 + band):
            side = "long"; reason = "fast_above_slow"
        elif fast < slow * (1.0 - band):
            side = "short"; reason = "fast_below_slow"

        # 8) Caution flag for very jumpy tape
        caution = "elevated_volatility" if atrp > wild else None

        return {
            "side": side,
            "reason": reason,
            "trail_pct": round(trail, 6),
            "caution": caution,
            # optional introspection fields (handy when debugging)
            "eff_fast": eff_fast,
            "eff_slow": eff_slow,
            "atrp": round(atrp, 6),
            "band": round(band, 6),
        }
