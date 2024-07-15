import logging
import yaml
from pathlib import Path
from importlib import resources
import colorlog


def gandlf_logger_setup(log_dir=None, config_path="logging_config.yaml"):
    """
    It sets up the logger. Reads from logging_config.
    If log_dir is None, the logs are flashed to console.
    Args:
        log_dir (str): dir path for saving the logs
        config_path (str): file path for the configuration

    """

    if log_dir == None:  # flash logs
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.root.setLevel(logging.DEBUG)
        logging.root.addHandler(console_handler)

    else:  # create the log file
        output_dir = Path(log_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with resources.open_text("GANDLF", config_path) as file:
            config_dict = yaml.safe_load(file)
            config_dict["handlers"]["rotatingFileHandler"]["filename"] = str(
                Path.joinpath(output_dir, "gandlf.log")
            )
            logging.config.dictConfig(config_dict)

    logging.captureWarnings(True)


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
