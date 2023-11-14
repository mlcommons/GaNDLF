"""If input folder to the MLCube does not contain a csv file that defines the data cases,
a custom entrypoint is needed to create a temporary csv file before calling GaNDLF's run command.
This script should expect the same arguments passed to the command `mlcube run --task infer`,
i.e. it should expect the inputs and outputs defined in `mlcube.yaml` in the `infer` task.
Note that the device argument will be set by gandlf_deploy (gandlf_deploy will run the entrypoint
with --device)."""

import os
import argparse
import sys


def create_csv(data_path):
    """A function that creates a ./data.csv file from input folder."""
    # Add your logic here
    raise NotImplementedError


def run_gandlf(output_path, device):
    """
    A function that calls GaNDLF's generate metrics command with the previously created csv.

    Args:
        output_path (str): The path to the output file/folder
        parameters_file (str): The path to the parameters file
    """
    exit_status = os.system(
        "python3.9 gandlf_run --train False "
        f"--device {device} --config /embedded_config.yml "
        f"--modeldir /embedded_model/ -i ./data.csv -o {output_path}"
    )
    exit_code = os.WEXITSTATUS(exit_status)
    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", metavar="", type=str, required=True)
    parser.add_argument("--output_path", metavar="", type=str, default=None)
    parser.add_argument(
        "--device", metavar="", type=str, required=True, choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    create_csv(args.data_path)
    run_gandlf(args.output_path, args.device)
