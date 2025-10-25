#!/usr/bin/env python3
"""
data/feeder.py
Thin wrapper that polls adapter.fetch_ticker() and keeps a rolling tick window.
Adds context awareness: funding rate, open interest, and volatility snapshots.

Use vol_snapshot for trading logic. It is an EMA of absolute returns as a fraction per tick.
Use vol_per_min for logs or cross checks. It rescales the EMA by the observed tick cadence.

Funding rate is the periodic payment between longs and shorts on a perpetual contract.
Open interest is the total number of outstanding contracts. Volatility snapshot here is a
smoothed average of recent absolute price changes as a percent.
"""

from __future__ import annotations

import asyncio
import time
import random
import collections
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Any, Tuple

@dataclass
class Tick:
    ts_ms: int
    price: float

class DataFeeder:
    def __init__(self, adapter, *, window:int=600, poll_sec:float=1.0, ctx_sec:float=30.0):
        self.adapter = adapter
        self.window = int(window)
        self.poll_sec = float(poll_sec)
        self.ctx_sec = float(ctx_sec)
        self._ticks: Deque[Tick] = collections.deque(maxlen=self.window)
        self._task: Optional[asyncio.Task] = None
        self._ctx_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # context values
        self._funding_rate: Optional[float] = None
        self._next_funding_ts: Optional[int] = None
        self._open_interest: Optional[float] = None
        self._last_ctx_fetch: float = 0.0
        # volatility running EMA (fraction per tick)
        self._ema_abs_ret: Optional[float] = None
        self._ema_alpha: float = 2.0 / (min(self.window, 300) + 1.0)  # stable default
        # symbol meta passthrough if adapter exposes it
        self._symbol_meta: Optional[Dict[str, Any]] = None

    def set_poll_sec(self, sec: float) -> None:
        self.poll_sec = max(0.1, float(sec))

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="data_feeder")
        self._ctx_task = asyncio.create_task(self._ctx_loop(), name="data_feeder_ctx")

    async def stop(self) -> None:
        self._stop.set()
        for t in (self._task, self._ctx_task):
            if t:
                t.cancel()
        for t in (self._task, self._ctx_task):
            if t:
                with _suppress(asyncio.CancelledError):
                    await t

    async def _loop(self) -> None:
        # short backfill
        for _ in range(3):
            try:
                t = await self.adapter.fetch_ticker()
                ts = int(t.get("ts") or t.get("timestamp") or int(time.time() * 1000))
                price = float(t["price"])
                self._push_tick(ts, price, allow_zero_seed=False)  # do not seed EMA with zero
            except Exception as e:
                print(f"[FEEDER] backfill tick error: {e}")
            await asyncio.sleep(0.1)

        while not self._stop.is_set():
            try:
                t = await self.adapter.fetch_ticker()
                ts = int(t.get("ts") or t.get("timestamp") or int(time.time()*1000))
                price = float(t["price"])
                self._push_tick(ts, price)
            except Exception as e:
                print(f"[FEEDER] poll error: {e}")
            # small jitter to avoid aliasing with exchange tick cadence
            jitter = random.uniform(-0.15, 0.15)
            await asyncio.sleep(max(0.05, self.poll_sec + jitter))

    def _push_tick(self, ts_ms: int, price: float, *, allow_zero_seed: bool = True) -> None:
        prev_price = self._ticks[-1].price if self._ticks else None
        self._ticks.append(Tick(ts_ms=ts_ms, price=price))
        if prev_price is not None:
            frac = abs(price - prev_price) / max(1e-12, prev_price)
            if self._ema_abs_ret is None:
                if frac == 0.0 and not allow_zero_seed:
                    return
                self._ema_abs_ret = frac
            else:
                a = self._ema_alpha
                self._ema_abs_ret = a * frac + (1.0 - a) * self._ema_abs_ret

    async def _ctx_loop(self) -> None:
        # fetch funding and open interest on interval with safe fallback
        while not self._stop.is_set():
            try:
                fr, nxt_ts, oi = await self.adapter.fetch_funding_and_oi()
                self._funding_rate = fr
                self._next_funding_ts = nxt_ts
                self._open_interest = oi
            except Exception as e:
                print(f"[FEEDER] context fetch error: {e}")

            # optional symbol precision or steps
            try:
                if hasattr(self.adapter, "get_symbol_meta"):
                    self._symbol_meta = await self.adapter.get_symbol_meta()  # may be sync or async
                elif hasattr(self.adapter, "symbol_meta"):
                    self._symbol_meta = getattr(self.adapter, "symbol_meta")
            except Exception:
                self._symbol_meta = None

            self._last_ctx_fetch = time.time()
            await asyncio.sleep(self.ctx_sec)

    def last_price(self) -> Optional[float]:
        return self._ticks[-1].price if self._ticks else None

    def prices(self, n:int) -> list[float]:
        if n <= 0:
            return []
        return [tk.price for tk in list(self._ticks)[-n:]]

    def snapshot(self, n:int=120) -> Dict[str, object]:
        arr = self.prices(n)
        return {"count": len(arr), "prices": arr, "last": arr[-1] if arr else None}

    def atr_pct(self, n:int=120) -> Optional[float]:
        rows = self.prices(n + 1)
        if len(rows) < 2:
            return None
        diffs = []
        for i in range(1, len(rows)):
            a = rows[i-1]; b = rows[i]
            diffs.append(abs(b - a) / max(1e-12, a))
        return (sum(diffs) / len(diffs)) if diffs else None

    def _effective_tpm(self, max_intervals: int = 60) -> Optional[float]:
        """
        Compute observed ticks per minute from recent timestamp gaps.
        Uses up to max_intervals recent gaps. Returns None if insufficient data.
        """
        if len(self._ticks) < 2:
            return None
        ticks = list(self._ticks)
        gaps = []
        # use last up to max_intervals gaps
        for i in range(len(ticks) - 1, max(0, len(ticks) - 1 - max_intervals), -1):
            a = ticks[i - 1].ts_ms
            b = ticks[i].ts_ms
            if b > a:
                gaps.append((b - a) / 1000.0)
        if not gaps:
            return None
        avg_gap_s = sum(gaps) / len(gaps)
        if avg_gap_s <= 0:
            return None
        return 60.0 / avg_gap_s

    def _vol_per_min_from_ema(self) -> Optional[float]:
        if self._ema_abs_ret is None:
            return None
        tpm = self._effective_tpm()
        if tpm is None:
            # fallback to configured cadence if observed is unavailable
            tpm = 60.0 / max(1e-6, self.poll_sec)
        return self._ema_abs_ret * tpm

    def _last_tick_ts(self) -> Optional[float]:
        if not self._ticks:
            return None
        return self._ticks[-1].ts_ms / 1000.0

    def _ticks_stale(self, now: float) -> bool:
        lt = self._last_tick_ts()
        if lt is None:
            return True
        return (now - lt) > (2.0 * self.poll_sec)

    def context(self) -> Dict[str, Any]:
        vol_ema = self._ema_abs_ret
        per_min = self._vol_per_min_from_ema()
        now = time.time()
        is_stale_ctx = (now - self._last_ctx_fetch) > (2.0 * self.ctx_sec)
        last_tick_ts = self._last_tick_ts()
        is_stale_ticks = self._ticks_stale(now)
        return {
            "funding_rate": self._funding_rate,
            "next_funding_ts": self._next_funding_ts,
            "open_interest": self._open_interest,
            "vol_snapshot": vol_ema,
            "vol_snapshot_frac_ema": vol_ema,
            "vol_per_min": per_min,
            "last_ctx_ts": self._last_ctx_fetch,
            "is_stale_ctx": is_stale_ctx,
            "last_tick_ts": last_tick_ts,
            "is_stale_ticks": is_stale_ticks,
            "tick_count": len(self._ticks),
            "symbol_meta": self._symbol_meta,
        }

class _suppress:
    def __init__(self, *exc):
        self.exc = exc
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return et is not None and issubclass(et, self.exc)

# Backward alias if other modules import MarketFeeder
MarketFeeder = DataFeeder
