import yaml
import pickle
import os


def recover_config(modelDir: str, outputFile: str) -> bool:
    """
    This function recovers the configuration file from a model directory.

    Args:
        modelDir (str): The model directory with the configuration file and the model.
        outputFile (str): The output file for the configuration.

    Returns:
        bool: True if the configuration file was successfully recovered.
    """
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
            "penalty_weights",
            "sampling_weights",
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
