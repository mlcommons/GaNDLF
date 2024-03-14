import logging

import click

from GANDLF.entrypoints import append_copyright_to_help

# import anonymizer command
# import run command
# import construct_csv command
# import collect_stats command
# import patch_miner command
# import preprocess command
from GANDLF.entrypoints.verify_install import new_way as verify_install_command
# import config_generator command
# import recover_config command
# import deploy command
# import optimize_model command
# import generate_metrics command
# import debug_info command
# import update_version command
from GANDLF import version


def setup_logging(loglevel):
    logging.basicConfig(level=loglevel.upper())


@click.group()
@click.version_option(version, "--version", "-v", message="GANDLF Version: %(version)s")
@click.option(
    "--loglevel",
    default="INFO",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.pass_context  # Pass the context to subcommands
@append_copyright_to_help
def gandlf(ctx, loglevel):
    """GANDLF command-line tool."""
    ctx.ensure_object(dict)
    ctx.obj["LOGLEVEL"] = loglevel
    setup_logging(loglevel)


# TODO: add anonymizer command
# TODO: add run command
# TODO: add construct-csv command
# TODO: add collect-stats command
# TODO: add path-miner command
# TODO: add preprocess command
gandlf.add_command(verify_install_command, "verify-install")
# TODO: add config-generator command
# TODO: add recover-config command
# TODO: add deploy command
# TODO: add optimize-model command
# TODO: add generate-metrics command
# TODO: add debug-info command
# TODO: add update-version command

if __name__ == "__main__":
    gandlf()
