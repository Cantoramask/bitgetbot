#!/usr/bin/env python3
"""
app.py
Entry point for the Bitget bot.

What this file does:
1) Loads defaults from environment (if available) and asks the user simple questions
   at startup: symbol, stake (USDT), leverage, and live/paper mode.
2) Sets up a rotating daily log file in ./logs and also prints to the console.
3) Tries to import and run the real Orchestrator from orchestrator.py.
   If that file (or its dependencies) is not ready yet, it runs a safe fallback
   "heartbeat" loop so you can launch the app today without errors.

Plain-English definitions:
- Environment variable: a named value stored outside your code (for example in a .env file)
  that your program can read for settings like API keys or default leverage.
- API key: a pair of secret values (key and secret) from the exchange that proves who you are.
- Async (asynchronous): a way to run tasks without blocking each other so the program stays responsive.

You can run:
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

# Optional: load .env if present, but do not fail if python-dotenv isn't installed yet.
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
    margin_mode: str = "cross"  # fixed for now; wiring for later
    intelligence_check_sec: int = 10  # default; coach may auto-tune later

    @staticmethod
    def from_inputs(
        symbol: str,
        stake_usdt: float,
        leverage: int,
        live: bool,
        margin_mode: str = None,
        intelligence_check_sec: int = None,
    ) -> "AppConfig":
        return AppConfig(
            symbol=_normalize_symbol(symbol or os.getenv("SYMBOL", "BTC/USDT:USDT")),
            stake_usdt=stake_usdt
            if stake_usdt is not None
            else float(os.getenv("BASE_USDT_PER_TRADE", "50")),
            leverage=leverage if leverage is not None else int(os.getenv("LEVERAGE", "5")),
            live=live if live is not None else os.getenv("LIVE", "false").lower() in ("1", "true", "y", "yes"),
            margin_mode=(margin_mode or os.getenv("MARGIN_MODE", "cross")).lower(),
            intelligence_check_sec=intelligence_check_sec
            if intelligence_check_sec is not None
            else int(os.getenv("INTELLIGENCE_CHECK_SEC", "10")),
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

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    # File
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
    s = sym.strip().upper()
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
    return s


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default if default is not None else "")


def _ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    default = "N" if default_no else "Y"
    val = input(f"{prompt} (y/N) [{default}]: ").strip().lower()
    if not val:
        return not default_no
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
            f"[CFG] symbol={self.cfg.symbol} stake={self.cfg.stake_usdt:.2f}USDT lev={self.cfg.leverage} live={self.cfg.live}"
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
        # Expect orchestrator.py to expose class Orchestrator(cfg, logger)
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
    banner = textwrap.dedent(
        """
        Bitget Bot — Starter
        --------------------
        This program will ask for a few settings and start the bot.
        Tip: You can put defaults in .env to skip typing next time.
        """
    ).strip()
    print(banner)

    # Defaults from env if present
    default_symbol = os.getenv("SYMBOL", "BTC/USDT:USDT")
    default_stake = os.getenv("BASE_USDT_PER_TRADE", "50")
    default_lev = os.getenv("LEVERAGE", "5")
    default_live_env = os.getenv("LIVE", "").lower() in ("1", "true", "y", "yes")

    symbol_in = _ask("Symbol", default_symbol)
    stake_in = _ask("USDT per trade", default_stake)
    lev_in = _ask("Leverage x", default_lev)
    live_in = _ask_yes_no("Live trading", default_no=not default_live_env)

    cfg = AppConfig.from_inputs(
        symbol=symbol_in,
        stake_usdt=_safe_float(stake_in, float(default_stake)),
        leverage=max(1, min(100, _safe_int(lev_in, int(default_lev)))),  # clamp just in case
        live=bool(live_in),
    )

    logger = _setup_logging()
    logger.info("[BOOT] Bitget bot app starting")
    logger.info(
        f"[CFG] symbol={cfg.symbol} stake={cfg.stake_usdt:.2f}USDT lev={cfg.leverage} live={cfg.live} margin={cfg.margin_mode}"
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
    """Small local suppress context to avoid importing contextlib just for one use."""

    def __init__(self, *exceptions):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.exceptions)


def main():
    # On Windows, asyncio may need this policy for signal handling; Linux is fine.
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
