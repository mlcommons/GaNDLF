"""If the predictions and labels inputs to the MLCube should come from different
mount points and can't be referred to in a single csv, a custom entrypoint is
needed to create a temporary csv file before calling GaNDLF's generate metrics command.
This script should expect the same arguments passed to the command `mlcube run --task evaluate`,
i.e. it should expect the inputs and outputs defined in `mlcube.yaml` in the `evaluate` task"""

import os
import argparse
import sys


def create_csv(predictions, labels):
    """A function that creates a ./data.csv file from input folders."""
    # Add your logic here
    raise NotImplementedError


def run_gandlf(output_file, config):
    """
    A function that calls GaNDLF's generate metrics command with the previously created csv.

    Args:
        output_file (str): The path to the output file/folder
        config (str): The path to the parameters file
    """
    exit_status = os.system(
        f"gandlf generate-metrics -c {config} -i ./data.csv -o {output_file}"
    )
    exit_code = os.WEXITSTATUS(exit_status)
    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="", type=str, required=True)
    parser.add_argument("--predictions", metavar="", type=str, required=True)
    parser.add_argument("--output-file", metavar="", type=str, default=None)
    parser.add_argument("--labels", metavar="", type=str, required=True)

    args = parser.parse_args()

    create_csv(args.predictions, args.labels)
    run_gandlf(args.output_file, args.config)
