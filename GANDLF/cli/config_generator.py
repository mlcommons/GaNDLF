import yaml
from typing import List, Optional, Union
from pathlib import Path
from copy import deepcopy


def generate_new_configs_from_key_and_value(
    base_config: dict,
    key: str,
    value: Union[str, list, int],
    upper_level_key: Optional[str] = None,
) -> List[dict]:
    """
    Generate new configs based on a base config and a strategy.

    Args:
        base_config (dict): The base configuration to generate new configs from.
        key (str): The key to change in the base config.
        value (Union[str, list, int]): The value to change the key to.
        upper_level_key (Optional[str]): The upper level key in the base config; useful for dict. Defaults to None.

    Returns:
        List[dict]: A list of new configs.
    """
    configs_to_return = []
    if key == "patch_size":
        for new_patch_size in value:
            new_config = deepcopy(base_config)
            new_config["patch_size"] = new_patch_size
            new_config[key] = new_patch_size
            configs_to_return.append(new_config)
    else:
        # extending via lists is easy enough
        if isinstance(value, list):
            for v in value:
                new_config = deepcopy(base_config)
                if upper_level_key is None:
                    new_config[key] = v
                else:
                    new_config[upper_level_key][key] = v
                configs_to_return.append(new_config)
        # dicts require a bit more effort
        elif isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, dict):
                    if not configs_to_return:
                        # first round of configs
                        configs_to_return = generate_new_configs_from_key_and_value(
                            base_config, k, v, key
                        )
                    else:
                        # subsequent rounds of configs
                        temp_configs_to_return = []
                        for config in configs_to_return:
                            temp_configs_to_return.extend(
                                generate_new_configs_from_key_and_value(
                                    config, k, v, key
                                )
                            )
                        configs_to_return.extend(temp_configs_to_return)
                else:
                    raise NotImplementedError("Nested dicts are not supported yet.")
    return configs_to_return


def remove_duplicates(configs_list: List[dict]) -> List[dict]:
    """
    Remove duplicate configs from a list of configs.

    Args:
        configs_list (list): A list of configs.

    Returns:
        List[dict]: A list of configs with duplicates removed.
    """
    configs_to_return = []
    for config in configs_list:
        if config not in configs_to_return:
            configs_to_return.append(config)
    return configs_to_return


def config_generator(
    base_config_path: str, strategy_path: str, output_dir: str
) -> None:
    """
    Main function that runs the training and inference.

    Args:
        base_config_path (str): Path to baseline configuration.
        strategy_path (str): Strategy to employ when creating new configurations.
        output_dir (str): The output directory.
    """
    base_config = None
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    strategy = None
    with open(strategy_path, "r") as f:
        strategy = yaml.safe_load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    new_configs_list = []

    for key, value in strategy.items():
        if not new_configs_list:
            # create first round of configs
            new_configs_list = generate_new_configs_from_key_and_value(
                base_config, key, value
            )
            new_configs_list = remove_duplicates(new_configs_list)
        else:
            # create new configs from the previous round of configs
            temp_configs_to_return = []
            for config in new_configs_list:
                temp_configs_to_return.extend(
                    generate_new_configs_from_key_and_value(config, key, value)
                )
            new_configs_list.extend(temp_configs_to_return)
            new_configs_list = remove_duplicates(new_configs_list)

    for i, config in enumerate(new_configs_list):
        with open(f"{output_dir}/config_{i}.yaml", "w") as f:
            yaml.dump(config, f)
