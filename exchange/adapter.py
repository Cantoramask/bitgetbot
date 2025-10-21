#!/usr/bin/env python3
# bitgetbot/exchange/adapter.py
"""
exchange/adapter.py
Bitget exchange adapter using ccxt for USDT perpetuals.

Plain English.
This file is the hands and feet. It talks to Bitget. It sets leverage and margin mode, fetches price, checks any open position, places market orders, and closes with reduce-only orders. It converts your USDT stake to the correct contract amount using the exchange precision and limits.

Definitions.
Precision is how many decimals the exchange allows for amounts or prices. Lot size is the smallest trade amount allowed. Reduce-only means the order can only decrease or close a position, never increase it.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any, Dict, Optional

import ccxt


class BitgetAdapter:
    def __init__(self, *, logger, symbol: str, leverage: int, margin_mode: str, live: bool):
        self.log = logger
        self.symbol = symbol  # normalised like BTC/USDT:USDT
        self.live = bool(live)
        self.leverage = int(leverage)
        self.margin_mode = (margin_mode or "cross").lower()

        api_key = os.getenv("BITGET_API_KEY", "")
        api_secret = os.getenv("BITGET_API_SECRET", "")
        api_pass = os.getenv("BITGET_API_PASSPHRASE", "")

        if self.live and (not api_key or not api_secret or not api_pass):
            raise RuntimeError("Missing Bitget API credentials in .env. Set BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE")

        self.ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_pass,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",     # USDT perpetuals
                "defaultSubType": "linear"
            }
        })
        self._markets = None
        self._market = None

    async def connect(self):
        await self._ensure_markets()
        await self._ensure_symbol_ready()
        self.log.info(f"[EX] Bitget ready live={self.live} lev={self.leverage} margin={self.margin_mode}")

    async def _ensure_markets(self):
        def _load():
            return self.ex.load_markets()
        self._markets = await asyncio.to_thread(_load)

    async def _ensure_symbol_ready(self):
        sym = self.symbol
        if sym not in self._markets:
            if sym.endswith("USDT") and "/" not in sym:
                sym = f"{sym[:-4]}/USDT:USDT"
            elif "/" in sym and not sym.endswith(":USDT"):
                sym = f"{sym}:USDT"
            self.symbol = sym
        self._market = self._markets[self.symbol]
        if self.live:
            try:
                await asyncio.to_thread(self.ex.set_leverage, self.leverage, self.symbol)
            except Exception as e:
                self.log.info(f"[EX] set_leverage warn: {e}")
            try:
                await asyncio.to_thread(self.ex.set_margin_mode, self.margin_mode, self.symbol)
            except Exception as e:
                self.log.info(f"[EX] set_margin_mode warn: {e}")

    async def fetch_ticker(self) -> Dict[str, Any]:
        def _fetch():
            return self.ex.fetch_ticker(self.symbol)
        t = await asyncio.to_thread(_fetch)
        px = float(t.get("last", t.get("close", 0.0)))
        if not px or not math.isfinite(px):
            raise RuntimeError("Bad price from ticker")
        return {"symbol": self.symbol, "price": px, "ts": t.get("timestamp", int(time.time() * 1000))}

    async def get_open_position(self) -> Optional[Dict[str, Any]]:
        def _pos():
            return self.ex.fetch_positions([self.symbol])
        try:
            rows = await asyncio.to_thread(_pos)
        except Exception as e:
            self.log.info(f"[EX] fetch_positions error: {e}")
            return None
        for r in rows or []:
            if r.get("symbol") != self.symbol:
                continue
            amt = float(r.get("contracts", r.get("amount", 0)) or 0)
            if abs(amt) < 1e-12:
                continue
            side = "long" if amt > 0 else "short"
            entry = float(r.get("entryPrice") or r.get("entry_price") or 0.0)
            lev = int(r.get("leverage") or self.leverage)
            size_usdt = abs(amt) * entry
            return {
                "symbol": self.symbol,
                "side": side,
                "size_usdt": float(size_usdt),
                "entry_price": float(entry),
                "leverage": lev,
                "contracts": abs(amt),
            }
        return None

    def _amount_from_usdt(self, usdt: float, price: float) -> float:
        amount = float(usdt) / max(1.0, float(price))
        try:
            # Use ccxt’s precision logic for this market/symbol
            amount = float(self.ex.amount_to_precision(self.symbol, amount))
        except Exception:
            # Fallback: treat precision.amount as a step size (e.g. 0.0001)
            prec = self._market["precision"].get("amount", None)
            if isinstance(prec, (int, float)) and prec > 0:
                step = float(prec)
                amount = math.floor(amount / step) * step
        min_amt = float(self._market["limits"]["amount"]["min"] or 0)
        if amount < min_amt:
            amount = min_amt
        return float(amount)

    async def place_order(self, side: str, usdt: float) -> Optional[Dict[str, Any]]:
        t = await self.fetch_ticker()
        price = float(t["price"])
        amount = self._amount_from_usdt(usdt, price)
        req_side = "buy" if side == "long" else "sell"

        if not self.live:
            return {
                "symbol": self.symbol,
                "side": side,
                "size_usdt": float(usdt),
                "entry_price": price,
                "leverage": self.leverage,
                "contracts": amount,
            }

        params = {"reduceOnly": False}
        def _order():
            return self.ex.create_order(self.symbol, "market", req_side, amount, None, params)
        try:
            o = await asyncio.to_thread(_order)
        except Exception as e:
            self.log.info(f"[EX] create_order error: {e}")
            return None

        filled = float(o.get("filled", amount) or amount)
        avg = float(o.get("average", price) or price)
        size_usdt = filled * avg
        return {
            "symbol": self.symbol,
            "side": side,
            "size_usdt": float(size_usdt),
            "entry_price": float(avg),
            "leverage": self.leverage,
            "contracts": filled,
            "order": o,
        }

    async def close_position(self, pos: Dict[str, Any]) -> Optional[float]:
        amt = float(pos.get("contracts") or 0.0)
        if amt <= 0:
            t = await self.fetch_ticker()
            amt = self._amount_from_usdt(pos["size_usdt"], t["price"])

        close_side = "sell" if pos["side"] == "long" else "buy"
        params = {"reduceOnly": True}

        if not self.live:
            t = await self.fetch_ticker()
            px = float(t["price"])
            direction = 1 if pos["side"] == "long" else -1
            pnl_pct = direction * (px - float(pos["entry_price"])) / max(1.0, float(pos["entry_price"]))
            return float(pos["size_usdt"]) * pnl_pct

        def _order():
            return self.ex.create_order(self.symbol, "market", close_side, amt, None, params)
        try:
            o = await asyncio.to_thread(_order)
        except Exception as e:
            self.log.info(f"[EX] close_order error: {e}")
            return None

        exit_px = float(o.get("average") or 0.0)
        if exit_px <= 0:
            t = await self.fetch_ticker()
            exit_px = float(t["price"])
        direction = 1 if pos["side"] == "long" else -1
        pnl_pct = direction * (exit_px - float(pos["entry_price"])) / max(1.0, float(pos["entry_price"]))
        return float(pos["size_usdt"]) * pnl_pct
