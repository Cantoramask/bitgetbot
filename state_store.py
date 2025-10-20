#!/usr/bin/env python3
"""
state_store.py
Tiny, safe persistence helpers for the Bitget bot.

Plain-English summary:
You get two simple tools.
StateStore saves and loads a small JSON snapshot (your bot's memory). Think of it as a notepad that you overwrite each time with the latest state.
Journal appends one JSON object per line to a .jsonl file (your event diary). Think of it as a timeline where each new event is written at the end.

New terms explained once.
Atomic write means we write to a temporary file first, then swap it into place in one step. If the power dies mid-write, you keep the last good file.
JSONL means JSON lines. Each line is a standalone JSON object. It is easy to read and append to.

Both classes create the data folder automatically. Paths are stable and predictable so other modules can share the same files.
"""

from __future__ import annotations

import json
import os
import time
import typing as _t
from dataclasses import dataclass, asdict


# ---------------------------
# Low-level atomic file helper
# ---------------------------
def _atomic_write(path: str, payload: str) -> None:
    """
    Write text to `path` atomically.
    We write to path.tmp then os.replace to the final path so it is crash-safe.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------
# StateStore: small JSON snapshot
# ---------------------------
@dataclass
class _Header:
    schema: str = "bitgetbot.state"
    version: int = 1
    saved_ts: float = 0.0  # epoch seconds


class StateStore:
    """
    Save and load a tiny JSON snapshot safely.

    Typical use:
        st = StateStore("orchestrator")   # file at data/state/orchestrator.json
        st.save({"position": None, "wins": 3})
        last = st.load()

    Files are human-readable and crash-safe thanks to atomic writes.
    """

    def __init__(self, name: str, base_folder: str = "data"):
        if not name or any(c in name for c in "/\\"):
            raise ValueError("StateStore name must be a simple identifier, e.g. 'orchestrator'")
        self.base_folder = base_folder
        self.folder = os.path.join(self.base_folder, "state")
        self.path = os.path.join(self.folder, f"{name}.json")
        os.makedirs(self.folder, exist_ok=True)

    def load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Backward compatible: accept plain dicts without header.
            if isinstance(raw, dict) and "_meta" in raw:
                return raw
            return {"_meta": asdict(_Header(schema="bitgetbot.state", version=1, saved_ts=0.0)), **raw}
        except FileNotFoundError:
            return {"_meta": asdict(_Header(schema="bitgetbot.state", version=1, saved_ts=0.0))}
        except Exception as e:
            # If the file is corrupt, keep a .bad copy and start fresh.
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass
            return {"_meta": asdict(_Header(schema="bitgetbot.state", version=1, saved_ts=0.0)), "error": f"load_failed: {e}"}

    def save(self, data: dict) -> None:
        meta = _Header(saved_ts=time.time())
        if "_meta" in data and isinstance(data["_meta"], dict):
            # Preserve any known fields, just update saved_ts
            data["_meta"]["saved_ts"] = meta.saved_ts
            payload = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            payload = json.dumps({"_meta": asdict(meta), **data}, ensure_ascii=False, indent=2)
        _atomic_write(self.path, payload)

    def update(self, patch: dict) -> dict:
        """
        Convenience method. Loads current state, shallow-updates keys from patch, saves, and returns the new dict.
        """
        data = self.load()
        data.update(patch or {})
        self.save(data)
        return data


# ---------------------------
# Journal: append-only JSONL with size guard
# ---------------------------
class Journal:
    """
    Append events as one JSON object per line.

    Typical use:
        j = Journal("journal")            # file at data/journal/journal.jsonl
        j.write("open", {"side": "long", "usdt": 50})
        j.rotate_keep_bytes(5_000_000)    # optional; keep last ~5 MB

    Rotate methods keep the most recent bytes and discard older lines to avoid unbounded growth.
    """

    def __init__(self, name: str = "journal", base_folder: str = "data"):
        if not name or any(c in name for c in "/\\"):
            raise ValueError("Journal name must be a simple identifier, e.g. 'journal'")
        self.base_folder = base_folder
        self.folder = os.path.join(self.base_folder, "journal")
        self.path = os.path.join(self.folder, f"{name}.jsonl")
        os.makedirs(self.folder, exist_ok=True)

    def write(self, event: str, payload: _t.Dict[str, _t.Any]) -> None:
        rec = {"ts": time.time(), "event": event, **(payload or {})}
        line = json.dumps(rec, ensure_ascii=False)
        # Plain append is fine; JSONL is resilient to partial writes, but we still fsync for safety.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rotate_keep_bytes(self, keep_bytes: int = 5_000_000) -> None:
        """
        If the file is larger than 2 * keep_bytes, keep only the last keep_bytes.
        This is byte-based and does not parse lines to stay fast and simple.
        """
        try:
            size = os.path.getsize(self.path)
        except FileNotFoundError:
            return
        if size <= keep_bytes * 2:
            return
        tmp = self.path + ".rot"
        with open(self.path, "rb") as src, open(tmp, "wb") as dst:
            # Copy the tail. If the cut happens mid-line, the next write will continue fine.
            src.seek(max(0, size - keep_bytes))
            dst.write(src.read())
        os.replace(tmp, self.path)

    def rotate_keep_days(self, days: int = 60) -> None:
        """
        Timestamp-based rotation stub for compatibility with earlier code.
        This implementation simply calls rotate_keep_bytes with a generous cap.
        """
        self.rotate_keep_bytes(keep_bytes=10_000_000)


# ---------------------------
# Minimal self-test
# ---------------------------
if __name__ == "__main__":
    ss = StateStore("demo_orchestrator")
    now_state = ss.update({"symbol": "BTC/USDT:USDT", "wins": 2, "losses": 1})
    print("Saved state keys:", list(now_state.keys()))

    j = Journal("demo")
    j.write("startup", {"msg": "hello"})
    j.write("open", {"side": "long", "usdt": 25})
    j.rotate_keep_bytes(1024 * 64)
    print("Wrote a couple of journal lines to", j.path)
