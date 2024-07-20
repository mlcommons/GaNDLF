import logging

import click
from .subcommands import cli_subcommands
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup
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
    # setup_logging(loglevel)
    logger_setup()


# registers subcommands: `gandlf anonymizer`, `gandlf run`, etc.
for command_name, command in cli_subcommands.items():
    gandlf.add_command(command, command_name)

if __name__ == "__main__":
    # pylint: disable=E1120
    gandlf()
