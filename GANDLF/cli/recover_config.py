import argparse
import os
import pickle
import yaml

from .copyright_message import copyrightMessage


def recover_config(modelDir, outputFile):
    assert os.path.exists(
        modelDir
    ), "The model directory does not appear to exist. Please check parameters."

    pickle_location = os.path.join(modelDir, "parameters.pkl")
    assert os.path.exists(
        pickle_location
    ), "The model does not appear to have a configuration file. Please check parameters."

    with open(pickle_location, "rb") as handle:
        parameters = pickle.load(handle)
        os.makedirs(os.path.dirname(os.path.realpath(outputFile)), exist_ok=True)

        # Remove a few problematic objects from the output
        # These cannot be safe_dumped to YAML (or present other problems).
        # To avoid this, try to use primitives and don't use integers as dict keys.
        removable_entries = [
            "output_dir",
            "model_dir_embedded",
            "training_data",
            "validation_data",
            "testing_data",
            "device",
            "subject_spacing",
            "weights",
            "class_weights",
        ]

        for entry in removable_entries:
            parameters.pop(entry, None)

        with open(outputFile, "w") as f:
            print(parameters)
            f.write(
                yaml.safe_dump(parameters, sort_keys=False, default_flow_style=False)
            )

    print(f"Config written to {outputFile}.")
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF_RecoverConfig",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Recovers a config file from a GaNDLF model. If used from within a deployed GaNDLF MLCube, attempts to extract the config from the embedded model.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m",
        "--modeldir",
        metavar="",
        default="",
        type=str,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "-c",
        "--mlcube",
        metavar="",
        type=str,
        help="Pass this option to attempt to extract the config from the embedded model in a GaNDLF MLCube (if any). Only useful in that context.",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        metavar="",
        type=str,
        help="Path to an output file where the config will be written.",
    )

    args = parser.parse_args()

    if args.mlcube:
        search_dir = "/embedded_model/"
    else:
        search_dir = args.modeldir

    result = recover_config(search_dir, args.outputFile)
    assert result, "Config file recovery failed."


if __name__ == "__main__":
    main()