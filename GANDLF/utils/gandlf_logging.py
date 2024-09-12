import logging
import yaml
from pathlib import Path
from importlib import resources
import tempfile
from GANDLF.utils import get_unique_timestamp
import sys


def _create_tmp_log_file():
    tmp_dir = Path(tempfile.gettempdir())
    log_dir = Path.joinpath(tmp_dir, ".gandlf")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path.joinpath(log_dir, get_unique_timestamp() + ".log")
    return log_file


def _create_log_file(log_file):
    log_file = Path(log_file)
    log_file.write_text("Starting GaNDLF logging session \n")


def _configure_logging_with_logfile(log_file, config_path):
    with resources.open_text("GANDLF", config_path) as file:
        config_dict = yaml.safe_load(file)
        config_dict["handlers"]["rotatingFileHandler"]["filename"] = str(log_file)
        logging.config.dictConfig(config_dict)


def gandlf_excepthook(exctype, value, tb):
    if issubclass(exctype, Exception):
        logging.exception("Uncaught exception", exc_info=(exctype, value, tb))
    else:
        sys.__excepthook__(exctype, value, tb)


def logger_setup(log_file=None, config_path="logging_config.yaml") -> None:
    """
    It sets up the logger. Reads from logging_config.

    Args:
        log_file (str): dir path for saving the logs, defaults to `None`, at which time logs are flushed to console.
        config_path (str): file path for the configuration

    """

    logging.captureWarnings(True)
    log_tmp_file = log_file
    if log_file is None:  # create tmp file
        log_tmp_file = _create_tmp_log_file()
    _create_log_file(log_tmp_file)
    _configure_logging_with_logfile(log_tmp_file, config_path)
    sys.excepthook = gandlf_excepthook
    logging.info(f"The logs are saved in {log_tmp_file}")


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
