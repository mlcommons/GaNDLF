import os
import argparse
import sys
import pandas as pd


def create_csv(data_path):
    """This function expects `data_path` to be structured in a certain way:
    it should contain a list of folders: each one contains a 'image.nii.gz'.
    """
    input_data = []
    for subjectID in os.listdir(data_path):
        folder_path = os.path.join(data_path, subjectID)
        if not os.path.isdir(folder_path):
            continue
        image_path = os.path.join(folder_path, "image.nii.gz")
        image_path = os.path.abspath(image_path)
        record = {"SubjectID": subjectID, "Channel_0": image_path}
        input_data.append(record)

    input_data_df = pd.DataFrame(input_data)
    input_data_df.to_csv("./data.csv", index=False)


def run_gandlf(output_path, model_dir, device):
    """
    A function that calls GaNDLF's run command with the previously created csv.

    Args:
        output_path (str): The path to the output file/folder
        model_dir (str): The path where model is stored
        device (str): device to run on (i.e. CPU or GPU)
    """
    exit_status = os.system(
        "gandlf run --infer "
        f"--device {device} --config /embedded_config.yml "
        f"--model-dir /embedded_model/ -i ./data.csv -o {output_path}"
    )
    exit_code = os.WEXITSTATUS(exit_status)
    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", metavar="", type=str, required=True)
    parser.add_argument("--output-path", metavar="", type=str, default=None)
    parser.add_argument("--model-dir", metavar="", type=str, default=None)
    parser.add_argument(
        "--device", metavar="", type=str, required=True, choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    create_csv(args.input_data)
    run_gandlf(args.output_path, args.model_dir, args.device)
