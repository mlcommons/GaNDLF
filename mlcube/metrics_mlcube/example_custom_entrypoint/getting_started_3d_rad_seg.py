import os
import argparse
import sys
import pandas as pd


def create_csv(predictions, labels):
    """This function expects `predictions` to be structured in a certain way:
    it should contain the following paths: `testing/<subID>/<subID>_seg.nii.gz`
    where <subID> is an integer not padded with zeros.`labels` is expected to have
    a list of folders, each containing a 'mask.nii.gz'. Names of the folders are subject
    IDs (integers padded with zeros to have three digit places)."""

    # read and parse expected labels
    labels_dict = {}
    for subjectID in os.listdir(labels):
        folder_path = os.path.join(labels, subjectID)
        if not os.path.isdir(folder_path):
            continue
        label_path = os.path.join(folder_path, "mask.nii.gz")
        labels_dict[int(subjectID)] = label_path

    # generate data input dict
    input_data = []
    for subjectID in os.listdir(os.path.join(predictions, "testing")):
        pred_path = os.path.join(
            predictions, "testing", subjectID, f"{subjectID}_seg.nii.gz"
        )
        pred_path = os.path.abspath(pred_path)
        label_path = labels_dict[int(subjectID)]
        label_path = os.path.abspath(label_path)
        prediction_record = {
            "SubjectID": subjectID,
            "Prediction": pred_path,
            "Target": label_path,
        }
        input_data.append(prediction_record)

    input_data_df = pd.DataFrame(input_data)
    input_data_df.to_csv("./data.csv", index=False)


def run_gandlf(output_path, parameters_file):
    """
    A function that calls GaNDLF's generate metrics command with the previously created csv.

    Args:
        output_path (str): The path to the output file/folder
        parameters_file (str): The path to the parameters file
    """
    exit_status = os.system(
        f"python3.9 gandlf_generateMetrics -c {parameters_file} -i ./data.csv -o {output_path}"
    )
    exit_code = os.WEXITSTATUS(exit_status)
    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters_file", metavar="", type=str, required=True)
    parser.add_argument("--predictions", metavar="", type=str, required=True)
    parser.add_argument("--output_path", metavar="", type=str, default=None)
    parser.add_argument("--labels", metavar="", type=str, required=True)

    args = parser.parse_args()

    create_csv(args.predictions, args.labels)
    run_gandlf(args.output_path, args.parameters_file)
