from typing import Optional, Union
from .config_manager import ConfigManager


def parseConfig(
    config_file_path: Union[str, dict], version_check_flag: Optional[bool] = True
) -> None:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    print(
        "WARNING: `GANDLF.parseConfig` will be deprecated in favor of `GANDLF.config_manager` in a future version."
    )
    return ConfigManager(config_file_path, version_check_flag)
