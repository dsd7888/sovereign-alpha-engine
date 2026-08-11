"""Central logger. Import everywhere as: ``from sovereign_alpha.utils.logger import logger``."""
import sys

from loguru import logger

from config.settings import LOG_DIR

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
)
logger.add(
    LOG_DIR / "engine_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",
)

__all__ = ["logger"]
