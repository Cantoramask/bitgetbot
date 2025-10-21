#!/usr/bin/env python3
"""
config/settings.py
Centralised defaults with env overrides. Optional convenience layer.
Existing modules can keep using os.getenv; this just provides a single place if you prefer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

def _bool(name: str, default: bool=False) -> bool:
    raw = os.getenv(name, str(default)).lower()
    return raw in ("1", "true", "y", "yes")

def _int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default

def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass
class Settings:
    symbol: str = os.getenv("SYMBOL", "BTC/USDT:USDT")
    stake_usdt: float = _float("BASE_USDT_PER_TRADE", 50.0)
    leverage: int = _int("LEVERAGE", 5)
    margin_mode: str = os.getenv("MARGIN_MODE", "cross")
    vol_profile: str = os.getenv("VOL_PROFILE", "Medium")
    live: bool = _bool("LIVE", False)
    advisor_enabled: bool = _bool("ADVISOR_ENABLED", False)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    intelligence_check_sec: int = _int("INTELLIGENCE_CHECK_SEC", 10)
    logs_dir: str = os.getenv("LOGS_DIR", "logs")
    data_dir: str = os.getenv("DATA_DIR", "data")
    # optional risk caps
    max_notional_usdt: float = _float("RISK_MAX_NOTIONAL_DEFAULT", 2000.0)
