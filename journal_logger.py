#!/usr/bin/env python3
"""
journal_logger.py
Scribe for the Bitget bot.

Plain-English summary:
This module gives you one place to record what happened. Whenever the bot does something,
you call JournalLogger. It writes a readable line to your console log and also appends a
structured JSON line to data/journal/events.jsonl. That JSON file is easy to analyse later.

New terms explained once:
JSONL means JSON Lines. Each line is a valid JSON object. It is perfect for logs because
you can append safely and tools can read one line at a time.

Typical use:
    from journal_logger import JournalLogger
    from state_store import Journal as JsonlJournal
    jl = JournalLogger(py_logger, JsonlJournal("events"))
    jl.heartbeat(status="ready")

You do not need to change your existing logging setup in app.py. Pass its logger in here.
"""

from __future__ import annotations

import logging
import time
import os
from typing import Any, Dict, Optional

try:
    # Use the shared JSONL writer from state_store.py so all modules write to the same place.
    from state_store import Journal as JsonlJournal
except Exception as _e:
    # Soft fallback if file not present yet. This keeps the bot runnable during staged build.
    class JsonlJournal:  # type: ignore
        def __init__(self, name: str = "events", base_folder: str = "data"):
            self.path = f"{base_folder}/{name}.jsonl"
        def write(self, event: str, payload: Dict[str, Any]) -> None:
            pass
        def rotate_keep_bytes(self, keep_bytes: int = 5_000_000) -> None:
            pass


class JournalLogger:
    """
    Bridges human-friendly logging and machine-friendly JSONL.

    Design notes:
    1) Console lines are short and readable.
    2) JSONL lines include the same information plus extra fields for later analysis.
    3) Methods are specific to the bot’s needs so callers do not have to build dicts each time.
    """

    def __init__(self, logger: logging.Logger, jsonl: Optional[JsonlJournal] = None):
        if not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger")
        self.log = logger
        self.jsonl = jsonl or JsonlJournal("events")
        # Minimal mode hides noisy advisory or tuning chatter
        self.minimal = os.getenv("LOG_MINIMAL", "false").lower() in ("1", "true", "y", "yes")
        # Throttle settings (seconds). You asked for ~30s unless an actual decision is made.
        try:
            self.hb_interval = int(float(os.getenv("HEARTBEAT_INTERVAL_SEC", "30")))
        except Exception:
            self.hb_interval = 30
        try:
            self.decide_interval = int(float(os.getenv("DECISION_INTERVAL_SEC", "30")))
        except Exception:
            self.decide_interval = 30
        # Internal timers and last-signatures for de-duplication
        self._last_hb_ts: float = 0.0
        self._last_decide_ts: float = 0.0
        self._last_decide_sig: Optional[str] = None

    # ------------
    # Low-level
    # ------------
    def _write(self, level: str, msg: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        payload = payload or {}
        line = f"{msg}"
        if level == "info":
            self.log.info(line)
        elif level == "warning":
            self.log.warning(line)
        elif level == "error":
            self.log.error(line)
        else:
            self.log.info(line)
        try:
            self.jsonl.write(event, {"level": level, **payload})
        except Exception:
            # Never let logging crash the bot
            pass

    # ------------
    # Public API
    # ------------
    def startup(self, cfg: Dict[str, Any]) -> None:
        msg = "[START] bot starting"
        self._write("info", msg, "startup", {"cfg": cfg})

    def shutdown(self) -> None:
        msg = "[STOP] bot stopping"
        self._write("info", msg, "shutdown", {})

    def heartbeat(self, **fields: Any) -> None:
        # Throttle heartbeats to every hb_interval seconds unless it's a notable status.
        notable = str(fields.get("status", "")).lower() in {
            "open", "close", "takeover", "emergency_exit", "trail_exit", "partial_close", "open_failed", "open_failed_after_flip"
        }
        now = time.time()
        if not notable and (now - self._last_hb_ts) < max(1, self.hb_interval):
            return
        self._last_hb_ts = now
        msg = "[HB] " + " ".join(f"{k}={fields[k]}" for k in sorted(fields))
        self._write("info", msg, "heartbeat", fields)

    def decision(self, side: str, reason: str, trail_pct: float, cautions: Optional[str] = None) -> None:
        if self.minimal:
            return
        # Clean one-line note
        note = cautions
        if isinstance(note, str):
            note = " ".join(note.split())
            if len(note) > 300:
                note = note[:300]
        # Build signature to avoid repeating identical "wait" spam
        sig = f"{side}|{reason}|{round(trail_pct,6)}|{note or ''}"
        now = time.time()
        is_wait = (side == "wait")
        # Log immediately if it's not 'wait' (a real decision), or the signature changed; otherwise throttle.
        if is_wait and sig == self._last_decide_sig and (now - self._last_decide_ts) < max(1, self.decide_interval):
            return
        self._last_decide_sig = sig
        self._last_decide_ts = now
        msg = f"[DECIDE] side={side} trail={trail_pct:.4%} why={reason}" + (f" note={note}" if note else "")
        self._write("info", msg, "decision", {"side": side, "trail_pct": trail_pct, "reason": reason, "cautions": note})

    def knob_change(self, name: str, old: Any, new: Any, why: str) -> None:
        if self.minimal:
            return
        msg = f"[KNOB] {name}: {old} -> {new} ({why})"
        self._write("info", msg, "knob_change", {"name": name, "old": old, "new": new, "why": why})

    def trade_open(self, symbol: str, side: str, usdt: float, price: float, lev: int, *, reason: Optional[str] = None) -> None:
        msg = f"[OPEN] {symbol} {side} {usdt:.2f}USDT @ {price:.2f} x{lev}" + (f" because {reason}" if reason else "")
        self._write("info", msg, "trade_open", {"symbol": symbol, "side": side, "usdt": usdt, "price": price, "lev": lev, "reason": reason})

    def trade_close(self, symbol: str, side: str, usdt: float, exit_price: float, pnl_usdt: float, wins: int, losses: int, success_rate: float, *, reason: Optional[str] = None) -> None:
        msg = f"[CLOSE] {symbol} {side} {usdt:.2f}USDT @ {exit_price:.2f} pnl={pnl_usdt:.2f} sr={success_rate:.2f} W{wins}/L{losses}" + (f" because {reason}" if reason else "")
        self._write(
            "info",
            msg,
            "trade_close",
            {
                "symbol": symbol,
                "side": side,
                "usdt": usdt,
                "exit_price": exit_price,
                "pnl_usdt": pnl_usdt,
                "wins": wins,
                "losses": losses,
                "success_rate": success_rate,
                "reason": reason,
            },
        )

    def partial_close(self, symbol: str, side: str, fraction: float, at_r: float, price: float, *, reason: Optional[str] = None) -> None:
        msg = f"[PARTIAL] {symbol} {side} {fraction:.0%} at {at_r:.1f}R @ {price:.2f}" + (f" because {reason}" if reason else "")
        self._write("info", msg, "partial_close", {"symbol": symbol, "side": side, "fraction": fraction, "r_mult": at_r, "price": price, "reason": reason})

    def advisor(self, allow: bool, confidence: float, note: Optional[str] = None) -> None:
        if self.minimal:
            return
        msg = f"[ADVISOR] allow={allow} conf={confidence:.2f}"
        self._write("info", msg, "advisor", {"allow": allow, "confidence": confidence, "note": note})

    def warn(self, note: str, **fields: Any) -> None:
        msg = f"[WARN] {note}"
        self._write("warning", msg, "warning", {"note": note, **fields})

    def error(self, note: str, **fields: Any) -> None:
        msg = f"[ERROR] {note}"
        self._write("error", msg, "error", {"note": note, **fields})

    def exception(self, err: Exception, where: str = "") -> None:
        cls = err.__class__.__name__
        msg = f"[EXC] {where} {cls}: {err}"
        self._write("error", msg, "exception", {"where": where, "error": f"{cls}: {err}"})

    def rotate(self, keep_bytes: int = 10_000_000) -> None:
        try:
            self.jsonl.rotate_keep_bytes(keep_bytes)
        except Exception:
            pass

# ---------------------------
# Minimal self-test
# ---------------------------
if __name__ == "__main__":
    import logging
    from state_store import Journal as JsonlJournal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    pylog = logging.getLogger("bitgetbot")
    jl = JournalLogger(pylog, JsonlJournal("events"))

    jl.startup({"symbol": "BTC/USDT:USDT", "lev": 5, "live": False})
    jl.heartbeat(price=100123.45, pos="none", intel_sec=10, trail_init=0.008, trail_tight=0.004)
    jl.decision(side="long", reason="trend up, pullback to MA", trail_pct=0.008)
    jl.knob_change("intelligence_sec", 10, 15, "volatility high")
    jl.trade_open("BTC/USDT:USDT", "long", 50.0, 100100.0, 5, reason="demo_open")
    jl.partial_close("BTC/USDT:USDT", "long", 0.3, 1.0, 100300.0, reason="demo_partial")
    jl.trade_close("BTC/USDT:USDT", "long", 50.0, 100250.0, 7.25, wins=3, losses=1, success_rate=0.75, reason="demo_close")
    jl.advisor(allow=True, confidence=0.82, note="trend intact")
    jl.warn("websocket hiccup", reconnect_in_sec=3)
    try:
        raise RuntimeError("demo failure")
    except Exception as e:
        jl.exception(e, where="selftest")
    jl.shutdown()
    jl.rotate(256 * 1024)
    print("Wrote JSONL events to data/journal/events.jsonl")
