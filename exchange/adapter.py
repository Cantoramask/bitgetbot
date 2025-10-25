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
from typing import Any, Dict, Optional, Tuple

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
        self._lev_verified_ts = 0.0
        # leverage backoff state to avoid noisy retries on failure
        self._lev_backoff_sec = 0.0
        self._lev_backoff_until = 0.0

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
            # try set margin mode first
            try:
                await asyncio.to_thread(self.ex.set_margin_mode, self.margin_mode, self.symbol)
            except Exception as e:
                self.log.info(f"[EX] set_margin_mode warn: {e}")

            # set leverage; if Bitget rejects, step down to the highest allowed
            try:
                await asyncio.to_thread(self.ex.set_leverage, self.leverage, self.symbol)
                self._lev_verified_ts = time.time()
                self._lev_backoff_sec = 0.0
                self._lev_backoff_until = 0.0
            except Exception as e:
                max_lev = int(self._market.get("limits", {}).get("leverage", {}).get("max") or self.leverage)
                probed = False
                for lev in range(min(self.leverage, max_lev), 0, -1):
                    try:
                        await asyncio.to_thread(self.ex.set_leverage, lev, self.symbol)
                        self.leverage = lev
                        probed = True
                        self._lev_verified_ts = time.time()
                        self._lev_backoff_sec = 0.0
                        self._lev_backoff_until = 0.0
                        self.log.info(f"[EX] leverage set to {lev} after adjust")
                        break
                    except Exception:
                        continue
                if not probed:
                    self.log.info(f"[EX] set_leverage failed for all attempts: {e}")
                    # start backoff since we failed at boot
                    self._lev_backoff_sec = max(self._lev_backoff_sec or 30.0, 30.0)
                    self._lev_backoff_until = time.time() + self._lev_backoff_sec

    async def fetch_ticker(self) -> Dict[str, Any]:
        def _fetch():
            return self.ex.fetch_ticker(self.symbol)
        t = await asyncio.to_thread(_fetch)
        px = float(t.get("last", t.get("close", 0.0)))
        if not px or not math.isfinite(px):
            raise RuntimeError("Bad price from ticker")
        return {"symbol": self.symbol, "price": px, "ts": t.get("timestamp", int(time.time() * 1000))}

    async def get_open_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Return the single open position for the given symbol, normalised for the orchestrator.
        Fields returned:
          symbol, side ("long"/"short"), size_usdt (notional), entry_price, leverage, contracts
        """
        _sym = symbol or self.symbol

        def _pos():
            # ccxt returns a list of position dicts
            return self.ex.fetch_positions([_sym])

        try:
            rows = await asyncio.to_thread(_pos)
        except Exception as e:
            self.log.info(f"[EX] fetch_positions error: {e}")
            return None

        def _parse(rows_in):
            for r in rows_in or []:
                try:
                    if str(r.get("symbol") or "") != _sym:
                        continue

                    amt = float(r.get("contracts", r.get("amount", 0)) or 0.0)
                    if abs(amt) < 1e-12:
                        continue

                    side = "long" if amt > 0 else "short"
                    entry = float(r.get("entryPrice") or r.get("entry_price") or r.get("avgPrice") or 0.0)
                    lev = int(r.get("leverage") or self.leverage)

                    px = float(entry)
                    if px <= 0:
                        try:
                            t = self.ex.fetch_ticker(_sym)
                            px = float(t.get("last") or t.get("close") or 0.0)
                        except Exception:
                            px = 0.0

                    size_usdt = abs(amt) * px

                    return {
                        "symbol": _sym,
                        "side": side,
                        "size_usdt": float(size_usdt),
                        "entry_price": float(entry),
                        "leverage": lev,
                        "contracts": float(abs(amt)),
                    }
                except Exception as inner:
                    self.log.info(f"[EX] parse_position error: {inner}")
                    continue
            return None

        pos = _parse(rows)
        # If entry is missing right after a takeover/open, retry once to stabilise
        if pos and (pos["entry_price"] <= 0 or pos["size_usdt"] <= 0):
            try:
                rows2 = await asyncio.to_thread(_pos)
                pos2 = _parse(rows2)
                if pos2:
                    pos = pos2
            except Exception:
                pass

        return pos

    async def get_all_open_positions(self) -> list[Dict[str, Any]]:
        """
        Return all open positions in normalised form.
        Each item has:
          symbol, side ("long"/"short"), size_usdt, entry_price, leverage, contracts
        """
        out: list[Dict[str, Any]] = []

        def _pos_all():
            return self.ex.fetch_positions()

        try:
            rows = await asyncio.to_thread(_pos_all)
        except Exception as e:
            self.log.info(f"[EX] fetch_positions(all) error: {e}")
            return out

        for r in rows or []:
            try:
                sym = str(r.get("symbol") or "")
                if not sym:
                    continue

                amt = float(r.get("contracts", r.get("amount", 0)) or 0.0)
                if abs(amt) < 1e-12:
                    continue

                side = "long" if amt > 0 else "short"
                entry = float(r.get("entryPrice") or r.get("entry_price") or r.get("avgPrice") or 0.0)
                lev = int(r.get("leverage") or self.leverage)

                px = float(entry)
                if px <= 0:
                    try:
                        t = self.ex.fetch_ticker(sym)
                        px = float(t.get("last") or t.get("close") or 0.0)
                    except Exception:
                        px = 0.0

                size_usdt = abs(amt) * px

                out.append({
                    "symbol": sym,
                    "side": side,
                    "size_usdt": float(size_usdt),
                    "entry_price": float(entry),
                    "leverage": lev,
                    "contracts": float(abs(amt)),
                })
            except Exception as inner:
                self.log.info(f"[EX] parse_position(all) error: {inner}")
                continue

        return out

    def _amount_from_usdt(self, usdt: float, price: float, *, strict_min: bool = False) -> Optional[float]:
        """
        Convert a USDT stake into contracts, respecting leverage, step, min and max.
        If strict_min is True and the computed amount cannot meet the min lot after rounding,
        return None so callers can decide how to react.
        """
        notional = float(usdt) * max(1, int(self.leverage))
        amount = notional / max(1e-12, float(price))  # tiny floor avoids divide-by-zero without bias

        # Exchange limits
        limits = self._market.get("limits", {}) or {}
        amt_limits = limits.get("amount", {}) or {}
        step = amt_limits.get("step")
        min_amt = float(amt_limits.get("min") or 0.0)
        max_amt = amt_limits.get("max")
        max_amt = float(max_amt) if max_amt is not None else None

        # First try ccxt helper
        try:
            amount = float(self.ex.amount_to_precision(self.symbol, amount))
        except Exception:
            # Step rounding
            if step is not None:
                try:
                    step_f = float(step)
                    if step_f > 0:
                        amount = math.floor(amount / step_f) * step_f
                except Exception:
                    pass
            else:
                # Fall back to decimal places in market.precision.amount
                prec = self._market.get("precision", {}).get("amount")
                if isinstance(prec, int) and prec >= 0:
                    factor = 10 ** int(prec)
                    amount = math.floor(amount * factor) / factor

        # Enforce min and max after rounding
        if amount < min_amt:
            if strict_min:
                self.log.info(f"[EX] amount below min lot after rounding stake={usdt} price={price} amt={amount} min_amt={min_amt}")
                return None
            amount = min_amt
        if max_amt is not None and amount > max_amt:
            # Cap to max and align to step if available
            amount = max_amt
            if step is not None:
                try:
                    step_f = float(step)
                    if step_f > 0:
                        amount = math.floor(amount / step_f) * step_f
                except Exception:
                    pass

        return float(amount)

    async def _ensure_leverage_applied(self) -> None:
        if not self.live:
            return

        now = time.time()

        # Respect backoff window if a previous attempt failed
        if self._lev_backoff_until and now < self._lev_backoff_until:
            return

        # Periodic refresh only if enough time since last verified
        if now - self._lev_verified_ts <= 30.0:
            return
        try:
            await asyncio.to_thread(self.ex.set_leverage, self.leverage, self.symbol)
            self._lev_verified_ts = now
            self._lev_backoff_sec = 0.0
            self._lev_backoff_until = 0.0
        except Exception as e:
            # Start or grow exponential backoff up to 15 minutes
            self._lev_backoff_sec = max(self._lev_backoff_sec * 2 if self._lev_backoff_sec else 30.0, 30.0)
            self._lev_backoff_sec = min(self._lev_backoff_sec, 900.0)
            self._lev_backoff_until = now + self._lev_backoff_sec
            self.log.info(f"[EX] reapply leverage warn: {e} (backing off {int(self._lev_backoff_sec)}s)")

    async def place_order(self, side: str, usdt: float) -> Optional[Dict[str, Any]]:
        await self._ensure_leverage_applied()
        t = await self.fetch_ticker()
        price = float(t["price"])
        amount = self._amount_from_usdt(usdt, price, strict_min=True)
        if amount is None or amount <= 0:
            # Too small to trade after rounding to exchange rules
            self.log.info(f"[EX] place_order rejected: stake too small for min lot side={side} stake={usdt} price={price}")
            return None

        req_side = "buy" if side == "long" else "sell"

        if not self.live:
            # Paper should mirror live: size_usdt equals contracts * price (≈ stake × leverage)
            return {
                "symbol": self.symbol,
                "side": side,
                "size_usdt": float(amount * price),
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
            try:
                o = await asyncio.to_thread(_order)
            except Exception:
                return None

        filled = float(o.get("filled", amount) or amount)
        if not math.isfinite(filled) or filled <= 0:
            filled = amount
        if "limits" in self._market and "amount" in self._market["limits"]:
            m = self._market["limits"]["amount"]
            if m.get("max") is not None:
                try:
                    filled = min(filled, float(m["max"]))
                except Exception:
                    pass
        avg = float(o.get("average", price) or price)
        cost = o.get("cost")
        try:
            size_usdt = float(cost) if cost is not None else filled * avg
        except Exception:
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
            # Estimation path for closes should not hard fail on min lot
            new_amt = self._amount_from_usdt(pos["size_usdt"], t["price"], strict_min=False)
            amt = float(new_amt or 0.0)

        close_side = "sell" if pos["side"] == "long" else "buy"
        params = {"reduceOnly": True}

        if not self.live:
            t = await self.fetch_ticker()
            px = float(t["price"])
            direction = 1 if pos["side"] == "long" else -1
            denom = max(1e-12, float(pos["entry_price"]))
            pnl_pct = direction * (px - float(pos["entry_price"])) / denom
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
        avg_entry = float(pos["entry_price"])
        if not math.isfinite(avg_entry) or avg_entry <= 0:
            return None
        denom = max(1e-12, avg_entry)
        pnl_pct = direction * (exit_px - avg_entry) / denom
        return float(pos["size_usdt"]) * pnl_pct

    async def close_position_fraction(self, pos: Dict[str, Any], fraction: float) -> bool:
        """
        Reduce-only partial close. fraction is 0..1.
        """
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0.0:
            return True
        amt = float(pos.get("contracts") or 0.0) * fraction
        if amt <= 0:
            return True

        close_side = "sell" if pos["side"] == "long" else "buy"
        params = {"reduceOnly": True}

        if not self.live:
            # In paper mode just say OK
            return True

        def _order():
            return self.ex.create_order(self.symbol, "market", close_side, amt, None, params)
        try:
            await asyncio.to_thread(_order)
            return True
        except Exception as e:
            self.log.info(f"[EX] partial_close error: {e}")
            return False

    async def fetch_funding_and_oi(self) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        """
        Try to fetch funding rate and open interest using ccxt where available.
        Returns (funding_rate as fraction, next_funding_ts ms, open_interest contracts or notional).
        On failure, returns Nones.
        """
        fr = None
        next_ts = None
        oi = None
        # Funding
        try:
            row = await asyncio.to_thread(self.ex.fetch_funding_rate, self.symbol)
            # typical unified keys: fundingRate as fraction, nextFundingTime ms
            fr = float(row.get("fundingRate")) if row and row.get("fundingRate") is not None else None
            nft = row.get("nextFundingTime") or row.get("info", {}).get("nextFundingTime") if isinstance(row.get("info", {}), dict) else None
            next_ts = int(nft) if nft else None
        except Exception:
            pass
        # Open Interest
        try:
            # if exchange supports it
            if hasattr(self.ex, "fetch_open_interest"):
                row2 = await asyncio.to_thread(self.ex.fetch_open_interest, self.symbol)
                # unified may use "openInterestAmount" or "openInterest"
                raw_oi = row2.get("openInterest") or row2.get("openInterestAmount")
                oi = float(raw_oi) if raw_oi is not None else None
        except Exception:
            pass
        return fr, next_ts, oi
