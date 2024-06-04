import logging
from logging import config
import yaml


def gandlf_logger_setup(
    logger_name, config_path="logging_config.yaml"
) -> logging.Logger:
    """
    It sets up the logger. Read from logging_config.

    Args:
        logger_name (str): logger name, the name should be the same in the logging_config
        config_path (str): file path for the configuration

    Returns:
        logging.Logger
    """
    with open(config_path, "r") as file:
        config1 = yaml.safe_load(file)
        logging.config.dictConfig(config1)

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
