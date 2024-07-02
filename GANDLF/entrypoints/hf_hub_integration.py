import click
from GANDLF.cli import copyrightMessage
from argparse import ArgumentParser, RawTextHelpFormatter


from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.download import DownloadCommand
from huggingface_hub.commands.env import EnvironmentCommand
from huggingface_hub.commands.lfs import LfsCommands
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.commands.tag import TagCommands
from huggingface_hub.commands.upload import UploadCommand
from huggingface_hub.commands.user import UserCommands

description = """Hugging Face Hub: Streamline model management with upload, download, and more\n\n"""


@click.command(
    context_settings=dict(ignore_unknown_options=True), add_help_option=False
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def new_way(args):
    """Hugging Face Hub: Streamline model management with upload, download, and more"""

    parser = ArgumentParser(
        "gandlf hf",
        usage="gandlf hf <command> [<args>]",
        description=description + copyrightMessage,
        formatter_class=RawTextHelpFormatter,
    )

    commands_parser = parser.add_subparsers(help="gandlf hf command helpers")

    EnvironmentCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)
    UploadCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    LfsCommands.register_subcommand(commands_parser)
    ScanCacheCommand.register_subcommand(commands_parser)
    DeleteCacheCommand.register_subcommand(commands_parser)
    TagCommands.register_subcommand(commands_parser)

    args = parser.parse_args(args)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    service = args.func(args)
    service.run()
