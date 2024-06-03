import logging
from logging import config
import yaml

def gandlf_logger(logger_name,config_path = "logging.yaml") ->logging.Logger:
    """
    It sets up the logger. Reads from the logging_config.

    Args:
        logger_name (str): logger name, the name should be the same in the logging_config
        config_path (str): file path for the configuration

    Returns:
        logging.Logger
    """
    with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.config.dictConfig(config)

    return logging.getLogger(logger_name)

