#!/usr/bin/env python3
"""
data/feeder.py
Thin wrapper that polls adapter.fetch_ticker() and keeps a rolling tick window.
Exposes latest price and small history for strategies.
"""

from __future__ import annotations

import asyncio
import time
import collections
from dataclasses import dataclass
from typing import Deque, Dict, Optional

@dataclass
class Tick:
    ts_ms: int
    price: float

class DataFeeder:
    def __init__(self, adapter, *, window:int=600, poll_sec:float=1.0):
        self.adapter = adapter
        self.window = int(window)
        self.poll_sec = float(poll_sec)
        self._ticks: Deque[Tick] = collections.deque(maxlen=self.window)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="data_feeder")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with _suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        # short backfill
        for _ in range(3):
            try:
                t = await self.adapter.fetch_ticker()
                self._ticks.append(Tick(ts_ms=int(t["ts"]), price=float(t["price"])))
            except Exception:
                pass
            await asyncio.sleep(0.1)

        while not self._stop.is_set():
            try:
                t = await self.adapter.fetch_ticker()
                ts = int(t.get("ts") or t.get("timestamp") or int(time.time()*1000))
                price = float(t["price"])
                self._ticks.append(Tick(ts_ms=ts, price=price))
            except Exception:
                # swallow, try next
                pass
            await asyncio.sleep(self.poll_sec)

    def last_price(self) -> Optional[float]:
        return self._ticks[-1].price if self._ticks else None

    def prices(self, n:int) -> list[float]:
        if n <= 0:
            return []
        return [tk.price for tk in list(self._ticks)[-n:]]

    def snapshot(self, n:int=120) -> Dict[str, object]:
        arr = self.prices(n)
        return {"count": len(arr), "prices": arr, "last": arr[-1] if arr else None}

class _suppress:
    def __init__(self, *exc):
        self.exc = exc
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return et is not None and issubclass(et, self.exc)
