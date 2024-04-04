from GANDLF.cli import copyrightMessage


# import anonymizer command
# import run command
from GANDLF.entrypoints.construct_csv import new_way as construct_csv_command

# import collect_stats command
# import patch_miner command
# import preprocess command
# import verify_install command
# import config_generator command
# import recover_config command
# import deploy command
# import optimize_model command
# import generate_metrics command
# import debug_info command
# import split_csv command


def append_copyright_to_help(command_func):
    command_func.__doc__ = (
        copyrightMessage
        if command_func.__doc__ is None
        else (command_func.__doc__ + "\n\n" + copyrightMessage)
    )
    return command_func


cli_subcommands = {
    # TODO: add anonymizer command
    # TODO: add run command
    "construct-csv": construct_csv_command,
    # TODO: add collect-stats command
    # TODO: add path-miner command
    # TODO: add preprocess command
    # TODO: add verify-install command
    # TODO: add config-generator command
    # TODO: add recover-config command
    # TODO: add deploy command
    # TODO: add optimize-model command
    # TODO: add generate-metrics command
    # TODO: add debug-info command
    # TODO: add split-csv command
}
