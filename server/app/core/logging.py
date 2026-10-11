from __future__ import annotations

import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """Routes stdlib records (uvicorn, sqlalchemy, livekit) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LogConfig:
    FORMAT = (
        "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>"
    )
    SILENCED = ("sqlalchemy.engine", "httpx", "httpcore")

    @classmethod
    def install(cls, level: str = "INFO", *, intercept: bool = True) -> None:
        """`intercept=False` where something else owns stdlib logging, such as the
        LiveKit CLI, which installs its own root handler and would double every line."""
        logger.remove()
        logger.add(sys.stderr, level=level, format=cls.FORMAT, backtrace=False, diagnose=False)
        if intercept:
            logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        for name in cls.SILENCED:
            logging.getLogger(name).setLevel(logging.WARNING)
