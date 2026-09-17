"""日志工具。"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "moe_rl",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """设置日志记录器。"""
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "moe_rl") -> logging.Logger:
    """获取已配置的日志记录器。"""
    return logging.getLogger(name)
