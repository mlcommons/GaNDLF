# import anonymizer command
# import run command
# import construct_csv command
# import collect_stats command
# import patch_miner command
from GANDLF.entrypoints.preprocess import new_way as preprocess_command

# import verify_install command
# import config_generator command
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
    "preprocess": preprocess_command,
    # TODO: add verify-install command
    # TODO: add config-generator command
    # TODO: add recover-config command
    # TODO: add deploy command
    # TODO: add optimize-model command
    # TODO: add generate-metrics command
    # TODO: add debug-info command
    # TODO: add split-csv command
}
