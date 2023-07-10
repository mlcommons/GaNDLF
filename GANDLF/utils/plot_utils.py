import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO


def plot_classification(
    df_training, df_validation, df_testing, output_plot, metrics=["accuracy"]
):
    banned_col_nametags = ["per_class", "per_label", "global"]

    # Get the columns that contain the banned nametags
    banned_cols = [
        col
        for col in df_training.columns
        if any(tag in col for tag in banned_col_nametags)
    ]

    # Split the values of the banned columns into multiple columns
    for df in [df_training, df_validation, df_testing]:
        for col in banned_cols:
            if df[col].dtype == "object":
                split_cols = (
                    df[col]
                    .str.split("_", expand=True)
                    .apply(pd.to_numeric, errors="coerce")
                )
                split_cols.columns = [f"{col}_{i}" for i in range(split_cols.shape[1])]
                df.drop(columns=col, inplace=True)
                df = pd.concat([df, split_cols], axis=1)

    # Check if any of the metrics is present in the column names of the dataframe
    assert any(
        any(metric in col for col in df_training.columns) for metric in metrics
    ), "None of the specified metrics is in the dataframe."

    required_cols = [
        "epoch_no",
        "train_loss",
    ]

    # Check if the required columns are in the dataframe
    assert all(
        col in df_training.columns for col in required_cols
    ), "Not all required columns are in the dataframe."

    epochs = len(df_training)

    # Plot for loss and metrics
    fig, axes = plt.subplots(
        nrows=1, ncols=len(metrics) + 1, figsize=((len(metrics) + 1) * 6, 6)
    )
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if "train_loss" in df_training.columns:
        sns.lineplot(
            data=df_training,
            x="epoch_no",
            y="train_loss",
            ax=axes[0],
            label="Training",
        )

    if "valid_loss" in df_validation.columns:
        sns.lineplot(
            data=df_validation,
            x="epoch_no",
            y="valid_loss",
            ax=axes[0],
            label="Validation",
        )

    if "test_loss" in df_testing.columns:
        sns.lineplot(
            data=df_testing,
            x="epoch_no",
            y="test_loss",
            ax=axes[0],
            label="Testing",
        )

    axes[0].set(xlim=(0, epochs - 1))
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss Plot")
    axes[0].legend()

    for i, metric in enumerate(metrics, start=1):
        metric_cols = [col for col in df_training.columns if metric in col]
        for metric_col in metric_cols:
            if metric_col in df_training.columns:
                sns.lineplot(
                    data=df_training,
                    x="epoch_no",
                    y=metric_col,
                    ax=axes[i],
                    label=f"Training {metric_col}",
                )
            if metric_col.replace("train", "valid") in df_validation.columns:
                sns.lineplot(
                    data=df_validation,
                    x="epoch_no",
                    y=metric_col.replace("train", "valid"),
                    ax=axes[i],
                    label=f"Validation {metric_col}",
                )
            if metric_col.replace("train", "test") in df_testing.columns:
                sns.lineplot(
                    data=df_testing,
                    x="epoch_no",
                    y=metric_col.replace("train", "test"),
                    ax=axes[i],
                    label=f"Testing {metric_col}",
                )
        axes[i].set(xlim=(0, epochs - 1))
        axes[i].set(
            xlabel="Epoch",
            ylabel=metric.capitalize(),
            title=f"{metric.capitalize()} Plot",
        )
        axes[i].legend()

    plt.savefig(output_plot, dpi=600)

    print("Classification plots saved successfully.")
    return df_training, df_validation, df_testing


def plot_segmentation(
    df_training, df_validation, df_testing, output_plot, metrics=["dice"]
):
    print("Segmentation task detected, generating dice and loss plots.")

    banned_col_nametags = ["per_class", "per_label", "global"]

    # Check if any columns in training and validation logs contain banned col nametags in any column name
    # Just the reference of the banned column in the name is enough to ban the entire column
    df_training = df_training.loc[
        :, ~df_training.columns.str.contains("|".join(banned_col_nametags))
    ]
    df_validation = df_validation.loc[
        :, ~df_validation.columns.str.contains("|".join(banned_col_nametags))
    ]

    # Assert that atleast one of the metrics is present in the column names of the dataframe
    assert any(metric in df_training.columns for metric in metrics)

    required_cols = [
        "epoch_no",
        "train_loss",
        "valid_loss",
    ]

    assert all(col in df_training.columns for col in required_cols)

    assert all(col in df_validation.columns for col in required_cols)

    epochs = len(df_training)

    # Try to plot for loss first
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    sns.lineplot(
        data=df_training,
        x="epoch_no",
        y="train_dice",
        ax=axes[0],
        label="Training",
    )
    sns.lineplot(
        data=df_validation,
        x="epoch_no",
        y="valid_dice",
        ax=axes[0],
        label="Validation",
    )
    axes[0].set(xlim=(0, epochs - 1), ylim=(0, 1))
    axes[0].set(xlabel="Epoch", ylabel="Dice", title="Dice Plot")
    axes[0].legend()

    sns.lineplot(
        data=df_training,
        x="epoch_no",
        y="train_dice",
        ax=axes[1],
        label="Training",
    )
    sns.lineplot(
        data=df_validation,
        x="epoch_no",
        y="valid_loss",
        ax=axes[1],
        label="Validation",
    )
    axes[1].set(xlim=(0, epochs - 1))
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss Plot")
    axes[1].legend()

    plt.savefig(output_plot, dpi=600)

    print("Classification plots saved successfully.")


def plot_regression(df_training, df_validation, df_testing):
    pass


def plot_synthesis(df_training, df_validation, df_testing):
    pass
