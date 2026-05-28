from __future__ import annotations

import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def silence(name: str) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL + 1)
