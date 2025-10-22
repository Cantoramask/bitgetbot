#!/usr/bin/env python3
"""
config/settings.py
Unified settings loader for the Bitget bot.

Why this exists:
- app.py expects a class named AppConfig with certain fields (including max_notional_usdt).
- Older modules may import Settings. We keep a backwards-compatible alias so nothing breaks.
- Automatically reads .env (if python-dotenv is present) so you don't have to 'load' it manually.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# --- Optional .env auto-load (safe if python-dotenv is not installed) ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass


# ---- helpers to read environment values safely ----
def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    try:
        return int(float(v.strip()))
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


# ---- MAIN CONFIG CLASS expected by app.py ----
@dataclass
class AppConfig:
    # Core trading
    symbol: str
    leverage: int
    margin_mode: str
    base_usdt_per_trade: float  # primary name going forward
    stake_usdt: float           # backward-compatible alias of the same value
    vol_profile: str
    live: bool
    takeover: bool

    # Behaviour & safety
    news_risk: bool
    risk_leverage_warn_x: int
    max_notional_usdt: float          # REQUIRED by app.py
    cross_exchange_check: bool

    # Intelligence & context
    intelligence_check_sec: int
    context_refresh_sec: int          # Item 1 fetch interval (funding/OI)

    # Partial take profits (Item 10)
    tp1_fraction: float
    tp2_fraction: float

    # Logging & paths
    logs_dir: str
    data_dir: str
    log_minimal: bool

    # Advisor / OpenAI (Item 2)
    advisor_enabled: bool
    openai_model: str
    openai_api_key: str

    # Bitget API keys
    bitget_api_key: str
    bitget_api_secret: str
    bitget_api_passphrase: str


def load_settings() -> AppConfig:
    """
    Build AppConfig from environment variables with safe defaults.
    This is what app.py should call.
    """

    # --- Core trading (matches your .env) ---
    symbol = _env_str("SYMBOL", "BTC/USDT:USDT")
    leverage = _env_int("LEVERAGE", 5)
    margin_mode = _env_str("MARGIN_MODE", "cross")
    base_usdt = _env_float("BASE_USDT_PER_TRADE", 50.0)
    vol_profile = _env_str("VOL_PROFILE", "auto")
    live = _env_bool("LIVE", False)
    takeover = _env_bool("TAKEOVER", False)

    # --- Behaviour & safety knobs ---
    news_risk = _env_bool("NEWS_RISK", False)
    risk_leverage_warn_x = _env_int("RISK_LEVERAGE_WARN_X", 30)
    # The “max budget” your launcher requires — previously named RISK_MAX_NOTIONAL_DEFAULT in .env
    max_notional_usdt = _env_float("RISK_MAX_NOTIONAL_DEFAULT", 5000.0)
    cross_exchange_check = _env_bool("CROSS_EXCHANGE_CHECK", False)

    # --- Intelligence & context (Item 1, Item 2 cadence) ---
    intelligence_check_sec = _env_int("INTELLIGENCE_CHECK_SEC", 10)
    context_refresh_sec = _env_int("CONTEXT_REFRESH_SEC", 30)

    # --- Partial take profits (Item 10) ---
    tp1_fraction = _env_float("TP1_FRACTION", 0.30)
    tp2_fraction = _env_float("TP2_FRACTION", 0.30)

    # --- Logging & paths ---
    logs_dir = _env_str("LOGS_DIR", "logs")
    data_dir = _env_str("DATA_DIR", "data")
    log_minimal = _env_bool("LOG_MINIMAL", True)

    # --- Advisor (Item 2) ---
    advisor_enabled = _env_bool("ADVISOR_ENABLED", False)
    openai_model = _env_str("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key = _env_str("OPENAI_API_KEY", "")

    # --- Bitget API keys ---
    bitget_api_key = _env_str("BITGET_API_KEY", "")
    bitget_api_secret = _env_str("BITGET_API_SECRET", "")
    bitget_api_passphrase = _env_str("BITGET_API_PASSPHRASE", "")

    return AppConfig(
        symbol=symbol,
        leverage=leverage,
        margin_mode=margin_mode,
        base_usdt_per_trade=base_usdt,
        stake_usdt=base_usdt,  # keep the old name available to older code
        vol_profile=vol_profile,
        live=live,
        takeover=takeover,
        news_risk=news_risk,
        risk_leverage_warn_x=risk_leverage_warn_x,
        max_notional_usdt=max_notional_usdt,
        cross_exchange_check=cross_exchange_check,
        intelligence_check_sec=intelligence_check_sec,
        context_refresh_sec=context_refresh_sec,
        tp1_fraction=tp1_fraction,
        tp2_fraction=tp2_fraction,
        logs_dir=logs_dir,
        data_dir=data_dir,
        log_minimal=log_minimal,
        advisor_enabled=advisor_enabled,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        bitget_api_key=bitget_api_key,
        bitget_api_secret=bitget_api_secret,
        bitget_api_passphrase=bitget_api_passphrase,
    )


# ---- Backwards compatibility for older imports ----
# Some modules may still do: `from config.settings import Settings`
Settings = AppConfig  # type: ignore
get_settings = load_settings  # old helper name if used anywhere
