import traceback
from typing import Union
import yaml
from pydantic import ValidationError

from GANDLF.configuration.parameters_config import Parameters
from GANDLF.configuration.exclude_parameters import exclude_parameters
from GANDLF.configuration.utils import handle_configuration_errors


def _parseConfig(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> None:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    params = config_file_path
    try:
        if not isinstance(config_file_path, dict):
            params = yaml.safe_load(open(config_file_path, "r"))
    except yaml.YAMLError as e:
        # this is a special case for config files with panoptica parameters
        from panoptica.utils.config import _load_yaml

        params = _load_yaml(config_file_path)

    return params


def ConfigManager(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> dict:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    try:
        parameters_config = Parameters(
            **_parseConfig(config_file_path, version_check_flag)
        )
        parameters = parameters_config.model_dump(
            exclude={
                field
                for field in exclude_parameters
                if getattr(parameters_config, field) is None
            }
        )
        return parameters

    except Exception as e:
        if isinstance(e, ValidationError):
            handle_configuration_errors(e)
            raise
        ## todo: ensure logging captures assertion errors
        else:
            assert (
                False
            ), f"Config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # logging.error(
        #     f"gandlf config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # )
        # raise
