# import anonymizer command
# import run command
# import construct_csv command
# import collect_stats command
# import patch_miner command
# import preprocess command
# import verify_install command
# import config_generator command
from GANDLF.entrypoints.recover_config import new_way as recover_config_command

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
    # TODO: add config-generator command
    "recover-config": recover_config_command,
    # TODO: add deploy command
    # TODO: add optimize-model command
    # TODO: add generate-metrics command
    # TODO: add debug-info command
    # TODO: add split-csv command
}
