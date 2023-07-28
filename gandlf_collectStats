#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from GANDLF.cli import copyrightMessage


def plot_all(df_training, df_validation, df_testing, output_plot_dir):
    """
    Plots training, validation, and testing data for loss and other metrics.
    TODO: this function needs to be moved under utils and then called after every training epoch.

    Args:
        df_training (pd.DataFrame): DataFrame containing training data.
        df_validation (pd.DataFrame): DataFrame containing validation data.
        df_testing (pd.DataFrame): DataFrame containing testing data.
        output_plot_dir (str): Directory to save the plots.

    Returns:
        tuple: Tuple containing the modified training, validation, and testing DataFrames.
    """
    # Drop any columns that might have "_" in the values of their rows
    banned_cols = [
        col
        for col in df_training.columns
        if any("_" in str(val) for val in df_training[col].values)
    ]

    # Determine metrics from the column names by removing the "train_" prefix
    metrics = [
        col.replace("train_", "")
        for col in df_training.columns
        if "train_" in col and col not in banned_cols
    ]

    # Split the values of the banned columns into multiple columns
    # for df in [df_training, df_validation, df_testing]:
    #     for col in banned_cols:
    #         if df[col].dtype == "object":
    #             split_cols = (
    #                 df[col]
    #                 .str.split("_", expand=True)
    #                 .apply(pd.to_numeric, errors="coerce")
    #             )
    #             split_cols.columns = [f"{col}_{i}" for i in range(split_cols.shape[1])]
    #             df.drop(columns=col, inplace=True)
    #             df = pd.concat([df, split_cols], axis=1)

    # Check if any of the metrics is present in the column names of the dataframe
    assert any(
        any(metric in col for col in df_training.columns) for metric in metrics
    ), "None of the specified metrics is in the dataframe."

    required_cols = ["epoch_no", "train_loss"]

    # Check if the required columns are in the dataframe
    assert all(
        col in df_training.columns for col in required_cols
    ), "Not all required columns are in the dataframe."

    epochs = len(df_training)

    # Plot for loss
    plt.figure(figsize=(12, 6))
    if "train_loss" in df_training.columns:
        sns.lineplot(data=df_training, x="epoch_no", y="train_loss", label="Training")

    if "valid_loss" in df_validation.columns:
        sns.lineplot(
            data=df_validation, x="epoch_no", y="valid_loss", label="Validation"
        )

    if df_testing is not None and "test_loss" in df_testing.columns:
        sns.lineplot(data=df_testing, x="epoch_no", y="test_loss", label="Testing")

    plt.xlim(0, epochs - 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.legend()
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_plot_dir, "loss_plot.png"), dpi=300)
    plt.close()

    # Plot for other metrics
    for metric in metrics:
        metric_cols = [col for col in df_training.columns if metric in col]
        for metric_col in metric_cols:
            plt.figure(figsize=(12, 6))
            if metric_col in df_training.columns:
                sns.lineplot(
                    data=df_training,
                    x="epoch_no",
                    y=metric_col,
                    label=f"Training {metric_col}",
                )
            if metric_col.replace("train", "valid") in df_validation.columns:
                sns.lineplot(
                    data=df_validation,
                    x="epoch_no",
                    y=metric_col.replace("train", "valid"),
                    label=f"Validation {metric_col}",
                )
            if (
                df_testing is not None
                and metric_col.replace("train", "test") in df_testing.columns
            ):
                sns.lineplot(
                    data=df_testing,
                    x="epoch_no",
                    y=metric_col.replace("train", "test"),
                    label=f"Testing {metric_col}",
                )
            plt.xlim(0, epochs - 1)
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(f"{metric.capitalize()} Plot")
            plt.legend()
            plt.savefig(os.path.join(output_plot_dir, f"{metric}_plot.png"), dpi=300)
            plt.close()

    print("Plots saved successfully.")
    return df_training, df_validation, df_testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GANDLF_CollectStats",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Collect statistics from different testing/validation combinations from output directory.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-m",
        "--modeldir",
        metavar="",
        type=str,
        help="Input directory which contains testing and validation models log files",
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        metavar="",
        type=str,
        help="Output directory to save stats and plot",
    )

    args = parser.parse_args()

    inputDir = os.path.normpath(args.modeldir)
    outputDir = os.path.normpath(args.outputdir)
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    outputFile = os.path.join(outputDir, "data.csv")  # data file name
    outputPlot = os.path.join(outputDir, "plot.png")  # plot file

    trainingLogs = os.path.join(inputDir, "logs_training.csv")
    validationLogs = os.path.join(inputDir, "logs_validation.csv")
    testingLogs = os.path.join(inputDir, "logs_testing.csv")

    # Read all the files
    df_training = pd.read_csv(trainingLogs)
    df_validation = pd.read_csv(validationLogs)
    df_testing = pd.read_csv(testingLogs) if os.path.isfile(testingLogs) else None

    # Check for metrics in columns and do tight plots
    plot_all(df_training, df_validation, df_testing, outputPlot)
