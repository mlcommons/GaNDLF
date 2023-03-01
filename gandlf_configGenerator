import argparse
from GANDLF.cli import config_generator, copyrightMessage


if __name__ == "__main__":
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

    config_generator(args.config, args.strategy, args.output)

    print("Finished.")
