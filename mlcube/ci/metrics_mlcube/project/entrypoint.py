import os
import argparse
import json
import yaml


def convert_json_to_yaml(tmp_json_output, output_file):
    with open(tmp_json_output) as f:
        results = json.load(f)
    with open(output_file, "w") as f:
        yaml.dump(results, f)
    os.remove(tmp_json_output)


def run_gandlf(predictions, labels, output_file, config):
    """
    A function that calls GaNDLF's generate metrics command.

    Args:
        predictions (str): The path to predictions folder. It must contain a "predictions.csv" file
        labels (str): The path to labels folder. It must contain a "targets.csv" file.
        output_file (str): The path to the output file/folder
        config (str): The path to the parameters file

    Note: If predictions and labels CSVs contain paths,
          those paths should be relative to the containing folder.
    """
    predictions_csv = os.path.join(predictions, "predictions.csv")
    labels_csv = os.path.join(labels, "targets.csv")

    output_folder = os.path.dirname(output_file)
    tmp_json_output = os.path.join(output_folder, "results.json")

    exit_status = os.system(
        f"gandlf generate-metrics -c {config} -i {labels_csv},{predictions_csv} -o {tmp_json_output}"
    )
    exit_code = os.WEXITSTATUS(exit_status)
    if exit_code != 0:
        raise RuntimeError(f"GaNDLF process failed with exit code {exit_code}")
    convert_json_to_yaml(tmp_json_output, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="", type=str, required=True)
    parser.add_argument("--predictions", metavar="", type=str, required=True)
    parser.add_argument("--output-file", metavar="", type=str, default=None)
    parser.add_argument("--labels", metavar="", type=str, required=True)

    args = parser.parse_args()

    run_gandlf(args.predictions, args.labels, args.output_file, args.config)
