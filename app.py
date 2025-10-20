#!/usr/bin/env python3
"""
app.py
Entry point for the Bitget bot.

What this file does:
1) Shows the exact startup prompts you requested, in the same order and format.
2) Loads defaults from environment if present so you can just press Enter.
3) Sets up a daily log file in ./logs and prints to the console.
4) Tries to run orchestrator.Orchestrator(cfg, logger). If not present yet,
   runs a harmless fallback heartbeat so you can launch today.

Plain-English definitions:
- Environment variable: a named value stored outside your code (for example in a .env file)
  that your program can read for settings like API keys or default leverage.
- API key: a pair of secret values (key and secret) from the exchange that proves who you are.
- Async means the program can wait for things like network without freezing the whole app.

Run:
    python app.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import textwrap
import logging
import signal
from dataclasses import dataclass

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class AppConfig:
    symbol: str
    stake_usdt: float
    leverage: int
    live: bool
    margin_mode: str
    vol_profile: str  # "High" | "Medium" | "Low"
    takeover: bool    # whether to take over an open position if detected
    intelligence_check_sec: int = 10

    @staticmethod
    def from_inputs(
        symbol: str,
        stake_usdt: float,
        leverage: int,
        live: bool,
        margin_mode: str,
        vol_profile: str,
        takeover: bool,
        intelligence_check_sec: int | None = None,
    ) -> "AppConfig":
        return AppConfig(
            symbol=_normalize_symbol(symbol or os.getenv("SYMBOL", "BTC/USDT:USDT")),
            stake_usdt=stake_usdt if stake_usdt is not None else float(os.getenv("BASE_USDT_PER_TRADE", "50")),
            leverage=leverage if leverage is not None else int(os.getenv("LEVERAGE", "5")),
            live=bool(live) if live is not None else os.getenv("LIVE", "false").lower() in ("1", "true", "y", "yes"),
            margin_mode=(margin_mode or os.getenv("MARGIN_MODE", "cross")).lower(),
            vol_profile=_normalize_vol(vol_profile or os.getenv("VOL_PROFILE", "Medium")),
            takeover=bool(takeover),
            intelligence_check_sec=intelligence_check_sec if intelligence_check_sec is not None else int(os.getenv("INTELLIGENCE_CHECK_SEC", "10")),
        )


# ---------------------------
# Simple console + file logger
# ---------------------------
def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    today = dt.datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join("logs", f"journal_{today}.log")

    logger = logging.getLogger("bitgetbot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------
# Input helpers
# ---------------------------
def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    # Accept "BTC", "BTC/USDT", "BTCUSDT", or "BTC/USDT:USDT"
    if s.endswith(":USDT") and "/" in s:
        return s
    if "/" in s and not s.endswith(":USDT"):
        return f"{s}:USDT"
    if s.endswith("USDT") and "/" not in s:
        base = s[:-4]
        return f"{base}/USDT:USDT"
    if s.isalpha():
        return f"{s}/USDT:USDT"
    return s or "BTC/USDT:USDT"


def _normalize_vol(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("h", "high"):
        return "High"
    if v in ("l", "low"):
        return "Low"
    return "Medium"


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default if default is not None else "")


def _ask_yes_no(prompt: str, default_yes: bool | None = None) -> bool:
    # Matches the visual you want: "(Y/N)" with optional [Y] or [N]
    yn = "Y/N"
    if default_yes is True:
        prompt_full = f"{prompt} ({yn}) [Y]: "
    elif default_yes is False:
        prompt_full = f"{prompt} ({yn}) [N]: "
    else:
        prompt_full = f"{prompt} ({yn}): "
    val = input(prompt_full).strip().lower()
    if not val and default_yes is not None:
        return bool(default_yes)
    return val in ("y", "yes", "1", "true")


def _safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _safe_int(s: str, default: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


# ---------------------------
# Import Orchestrator or make a safe fallback
# ---------------------------
class _FallbackOrchestrator:
    """
    If orchestrator.py is not ready yet, we provide a harmless loop that
    logs a heartbeat every few seconds so the app can be started today.
    """

    def __init__(self, cfg: AppConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._stop = asyncio.Event()

    async def run(self):
        self.logger.info("[START] Fallback mode active (orchestrator not ready). No trading will occur.")
        self.logger.info(
            f"[CFG] symbol={self.cfg.symbol} stake={self.cfg.stake_usdt:.2f}USDT lev={self.cfg.leverage} "
            f"live={self.cfg.live} margin={self.cfg.margin_mode} vol={self.cfg.vol_profile} takeover={self.cfg.takeover}"
        )
        self.logger.info("When orchestrator.py is implemented, the real trading loop will run here.")
        try:
            while not self._stop.is_set():
                self.logger.info("[HB] waiting for orchestrator module... press Ctrl+C to stop")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        self.logger.info("[STOP] Fallback orchestrator stopped.")

    def request_stop(self):
        self._stop.set()


def _load_orchestrator(cfg: AppConfig, logger: logging.Logger):
    try:
        import importlib

        orch_module = importlib.import_module("orchestrator")
        Orchestrator = getattr(orch_module, "Orchestrator", None)
        if Orchestrator is None:
            raise AttributeError("orchestrator.Orchestrator not found")
        return Orchestrator(cfg, logger)
    except Exception as e:
        logger.info(f"[INFO] Using fallback orchestrator: {e}")
        return _FallbackOrchestrator(cfg, logger)


# ---------------------------
# Graceful shutdown
# ---------------------------
def _install_signal_handlers(loop, stopper):
    def _handle_sig():
        try:
            stopper()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # Windows may not support this fully; ignore.
            pass


# ---------------------------
# Main
# ---------------------------
async def _async_main():
    # Resolve default symbol before the takeover prompt so it can appear in parentheses
    default_symbol = _normalize_symbol(os.getenv("SYMBOL", "BTC/USDT:USDT"))
    default_stake = os.getenv("BASE_USDT_PER_TRADE", "50.0")
    default_lev = os.getenv("LEVERAGE", "5")
    default_live_env = os.getenv("LIVE", "").lower() in ("1", "true", "y", "yes")
    default_margin = os.getenv("MARGIN_MODE", "cross")
    default_vol = _normalize_vol(os.getenv("VOL_PROFILE", "Medium"))

    # 1) Takeover prompt exactly as requested
    takeover = _ask_yes_no(
        f"Open perpetual positions detected ({default_symbol}). Take over management now using existing size and leverage?",
        default_yes=None  # no [Y] or [N] displayed on this one
    )

    # 2) Symbol format examples block (pure print, no input yet)
    examples = textwrap.dedent(
        """\
        Symbol format examples:
          BTC/USDT:USDT  Bitget perpetuals
          BTC/USDT       auto adds :USDT
          BTCUSDT        converts to BTC/USDT:USDT
          BTC            converts to BTC/USDT:USDT"""
    )
    print(examples)

    # 3) Symbol [BTC/USDT:USDT]:
    symbol_in = _ask("Symbol", default_symbol)

    # 4) Live trading? (Y/N) [N]:
    live_in = _ask_yes_no("Live trading?", default_yes=False if not default_live_env else True)

    # 5) USDT margin per trade [50.0]:
    stake_in = _ask("USDT margin per trade", default_stake)

    # 6) Leverage x [5]:
    lev_in = _ask("Leverage x", default_lev)

    # 7) Margin mode cross or isolated [cross]:
    margin_in = _ask("Margin mode cross or isolated", default_margin)

    # 8) Volatility profile [High/Medium/Low] (default Medium):
    # User can type High, Medium, or Low; default is Medium when blank.
    vol_in = input("Volatility profile [High/Medium/Low] (default Medium): ").strip() or default_vol

    # Build config
    cfg = AppConfig.from_inputs(
        symbol=symbol_in,
        stake_usdt=_safe_float(stake_in, float(default_stake)),
        leverage=max(1, min(100, _safe_int(lev_in, int(default_lev)))),
        live=bool(live_in),
        margin_mode=margin_in,
        vol_profile=vol_in,
        takeover=bool(takeover),
    )

    logger = _setup_logging()
    logger.info("[BOOT] Bitget bot app starting")
    logger.info(
        f"[CFG] symbol={cfg.symbol} stake={cfg.stake_usdt:.2f}USDT lev={cfg.leverage} live={cfg.live} "
        f"margin={cfg.margin_mode} vol={cfg.vol_profile} takeover={cfg.takeover}"
    )

    orchestrator = _load_orchestrator(cfg, logger)

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()

    def _stop_all():
        try:
            if hasattr(orchestrator, "request_stop"):
                orchestrator.request_stop()
        except Exception:
            pass
        stopping.set()

    _install_signal_handlers(loop, _stop_all)

    # Run orchestrator until stop requested
    main_task = asyncio.create_task(orchestrator.run())
    await stopping.wait()
    main_task.cancel()
    with contextlib_suppress(asyncio.CancelledError):
        await main_task

    logger.info("[EXIT] Bye")


class contextlib_suppress:
    """Small local suppress context to avoid importing contextlib for one use."""

    def __init__(self, *exceptions):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.exceptions)


def main():
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
