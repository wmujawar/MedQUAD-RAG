import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from src.config import get_settings


def setup_logging() -> None:
    """
    Configure structlog processors
    """

    settings = get_settings()
    log_dir = "logs"

    Path(log_dir).mkdir(exist_ok=True)

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(message)s",
        handlers=[
            RotatingFileHandler(
                filename=f"{log_dir}/app.log",
                maxBytes=10 * 1024 * 1024,  # 10MB per file
                backupCount=5,  # Keep 5 backup files
                encoding="utf-8",
            )
        ],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.EventRenamer("message"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            min_level=getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
