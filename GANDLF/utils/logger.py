import logging, os, warnings
from typing import Optional, Tuple
from pathlib import Path

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """
    This function formats warning message according to its type
    """
    if category == UserWarning:
        return str(message)
    else:
        return '%s:%s: %s:%s' % (filename, lineno, category.__name__, message)


def setup_logger(output_dir: str, verbose: Optional[bool] = False) -> Tuple[logging.Logger, str, str]:
    """
    This function setups a logger with severity level controlled by verbose parameter from a config file.

    Args:
        logger_name (str): Name for a logger
        logs_dir (str): Output directory for log files.
        verbose (Optional[bool], optional): Used to setup the logging level. Defaults to False.

    Returns:
        logger (logging.Logger)
        logger_dir (str): directory for the logs
        logger_name (str): name of the logger
    """
    logs_dir = f'{output_dir}/logs'
    logger_name = 'gandlf'
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    warnings.formatwarning = warning_on_one_line
    logging.captureWarnings(True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs messages with severity determined by verbose param
    fh = logging.FileHandler(os.path.join(logs_dir, "gandlf.log"))
    fh.setLevel(logging.DEBUG) if verbose else fh.setLevel(logging.WARNING)
    warnings_logger = logging.getLogger("py.warnings")

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(fh)
    warnings_logger.addHandler(fh)

    return logger, logs_dir, logger_name
