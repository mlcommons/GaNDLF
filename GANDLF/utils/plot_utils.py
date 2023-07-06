import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO


def plot_classification(
    df_training, df_validation, df_testing, output_plot, combined_plots
):
    banned_col_nametags = ["per_class", "per_label", "global"]

    # Check if any columns in training and validation logs contain banned col nametags in any column name
    # Just the reference of the banned column in the name is enough to ban the entire column
    df_training = df_training.loc[
        :, ~df_training.columns.str.contains("|".join(banned_col_nametags))
    ]
    df_validation = df_validation.loc[
        :, ~df_validation.columns.str.contains("|".join(banned_col_nametags))
    ]

    required_cols = [
        "epoch_no",
        "train_loss",
        "valid_loss",
    ]

    if not all(col in df_training.columns for col in required_cols):
        raise ValueError(
            "Some required columns are missing in the training logs CSV file."
        )

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError(
            "Some required columns are missing in the validation logs CSV file."
        )

    epochs = len(df_training)

    # Try to plot for loss first
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    sns.lineplot(
        data=df_training,
        x="epoch_no",
        y="train_metric",
        ax=axes[0],
        label="Training",
    )
    sns.lineplot(
        data=df_validation,
        x="epoch_no",
        y="valid_metric",
        ax=axes[0],
        label="Validation",
    )
    axes[0].set(xlim=(0, epochs - 1), ylim=(0, 1))
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Plot")
    axes[0].legend()

    sns.lineplot(
        data=df_training,
        x="epoch_no",
        y="train_loss",
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


def plot_segmentation(df_training, df_validation, df_testing, output_plot):
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

    required_cols = [
        "epoch_no",
        "train_loss",
        "valid_loss",
    ]

    if not all(col in df_training.columns for col in required_cols):
        raise ValueError(
            "Some required columns are missing in the training logs CSV file."
        )

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError(
            "Some required columns are missing in the validation logs CSV file."
        )

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
