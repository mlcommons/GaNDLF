import logging
from logging import config
import yaml
from pathlib import Path

logging_config = {
    "version": 1,
    "formatters": {
        "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "filters": {
        "warnings_filter": {"()": "logging.Filter", "name": "py.warnings"},
        "info_only_filter": {"()": "gandlf_logger.InfoOnlyFilter"},
    },
    "handlers": {
        "stdoutHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filters": ["info_only_filter"],
            "stream": "ext://sys.stdout",
        },
        "stderrHandler": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "detailed",
            "stream": "ext://sys.stderr",
        },
        "debugHandler": {
            "class": "logging.FileHandler",
            "filename": "gandlf.log",
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "debug_logger": {
            "level": "DEBUG",
            "handlers": ["stdoutHandler", "debugHandler", "stderrHandler"],
            "propagate": False,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["stdoutHandler", "debugHandler", "stderrHandler"],
    },
}


def gandlf_logger_setup(logger_name) -> logging.Logger:
    """
    It sets up the logger. Read from logging_config.

    Args:
        logger_name (str): logger name, the name should be the same in the logging_config
        config_filename (str): file path for the configuration

    Returns:
        logging.Logger
    """
    logging.config.dictConfig(logging_config)

    logging.captureWarnings(True)

    return logging.getLogger(logger_name)


class InfoOnlyFilter(logging.Filter):
    """
    Display only INFO messages.
    """

    def filter(self, record):
        """
        Determines if the specified record is to be logged.

        Args:
            record (logging.LogRecord): The log record to be evaluated.

        Returns:
            bool: True if the log record should be processed, False otherwise.
        """
        return record.levelno == logging.INFO
