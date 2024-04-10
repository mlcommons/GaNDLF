# import anonymizer command
# import run command
# import construct_csv command
# import collect_stats command
# import patch_miner command
# import preprocess command
# import verify_install command
# import config_generator command
# import recover_config command
# import deploy command
# import optimize_model command
from GANDLF.entrypoints.generate_metrics import new_way as generate_metrics_command

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
    # TODO: add recover-config command
    # TODO: add deploy command
    # TODO: add optimize-model command
    "generate-metrics": generate_metrics_command,
    # TODO: add debug-info command
    # TODO: add split-csv command
}
