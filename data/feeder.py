#!/usr/bin/env python3
"""
data/feeder.py
Thin wrapper that polls adapter.fetch_ticker() and keeps a rolling tick window.
Adds context awareness: funding rate, open interest, and a simple volatility snapshot.

Definitions.
Funding rate is the periodic payment between longs and shorts on a perpetual contract.
Open interest is the total number of outstanding contracts. Volatility snapshot here is a
simple average of recent absolute price changes as a percent.
"""

from __future__ import annotations

import asyncio
import time
import collections
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Any

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
                self._ticks.append(Tick(ts_ms=int(t["ts"]), price=float(t["price"])))
            except Exception as e:
                print(f"[FEEDER] backfill tick error: {e}")
            await asyncio.sleep(0.1)

        while not self._stop.is_set():
            try:
                t = await self.adapter.fetch_ticker()
                ts = int(t.get("ts") or t.get("timestamp") or int(time.time()*1000))
                price = float(t["price"])
                self._ticks.append(Tick(ts_ms=ts, price=price))
            except Exception as e:
                print(f"[FEEDER] poll error: {e}")
            await asyncio.sleep(self.poll_sec)

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
            diffs.append(abs(b - a) / max(1.0, a))
        return (sum(diffs) / len(diffs)) if diffs else None

    def context(self) -> Dict[str, Any]:
        vol = self.atr_pct(120)
        return {
            "funding_rate": self._funding_rate,
            "next_funding_ts": self._next_funding_ts,
            "open_interest": self._open_interest,
            "vol_snapshot": vol,
            "last_ctx_ts": self._last_ctx_fetch,
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
