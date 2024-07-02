import logging
from logging import config
import yaml
from pathlib import Path
from importlib import resources


def gandlf_logger_setup(logger_name,config_path = "logging_config.yaml") -> logging.Logger:
    """
    It sets up the logger. Read from logging_config.
    Args:
        logger_name (str): logger name, the name should be the same in the logging_config
        config_path (str): file path for the configuration
    Returns:
        logging.Logger
    """

    # if config_path == None:
    #     config_dir = Path.cwd()
    #     config_path = Path.joinpath(config_dir, "GANDLF/config_gandlf_logger.yaml")

    # create dir for storing the messages
    current_dir = Path.cwd()
    directory = Path.joinpath(current_dir, "tmp/gandlf")
    directory.mkdir(parents=True, exist_ok=True)

    with resources.open_text("GANDLF", config_path) as file:
        config_dict = yaml.safe_load(file)
        logging.config.dictConfig(config_dict)

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