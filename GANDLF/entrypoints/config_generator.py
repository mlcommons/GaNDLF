import argparse
from deprecated import deprecated
import click

from GANDLF.cli import config_generator, copyrightMessage
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _generate_config(config: str, strategy: str, output: str):
    config_generator(config, strategy, output)
    print("Finished.")


@click.command()
@click.option(
    "--config",
    "-c",
    help="Path to base config.",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--strategy",
    "-s",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Config creation strategy in a yaml format.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to output directory.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(config, strategy, output, log_file):
    """Generate multiple GaNDLF configurations based on a single baseline GaNDLF for experimentation."""

    logger_setup(log_file)
    _generate_config(config, strategy, output)


# old-fashioned way of running gandlf via `gandlf_configGenerator`.
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf config-generator` cli command "
    + "instead of `gandlf_configGenerator`.\n"
    + "`gandlf_configGenerator` script would be deprecated soon."
)
def old_way():
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="GANDLF_ConfigGenerator",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate multiple GaNDLF configurations based on a single baseline GaNDLF for experimentation.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        help="Path to base config.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--strategy",
        metavar="",
        type=str,
        help="Config creation strategy in a yaml format.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        type=str,
        help="Path to output directory.",
        required=True,
    )

    args = parser.parse_args()

    _generate_config(args.config, args.strategy, args.output)
