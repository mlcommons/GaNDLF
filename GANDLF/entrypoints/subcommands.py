# import anonymizer command
# import run command
# import construct_csv command
# import collect_stats command
# import patch_miner command
# import preprocess command
# import verify_install command
from GANDLF.entrypoints.config_generator import new_way as config_generator_command

# import recover_config command
# import deploy command
# import optimize_model command
# import generate_metrics command
# import debug_info command
# import split_csv command

cli_subcommands = {
    # TODO: add anonymizer command
    # TODO: add run command
    # TODO: add construct-csv command
    # TODO: add collect-stats command
    # TODO: add path-miner command
    # TODO: add preprocess command
    # TODO: add verify-install command
    "config-generator": config_generator_command,
    # TODO: add recover-config command
    # TODO: add deploy command
    # TODO: add optimize-model command
    # TODO: add generate-metrics command
    # TODO: add debug-info command
    # TODO: add split-csv command
}
