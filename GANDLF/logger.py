#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:44 2021

@author: siddhesh
"""

import os
from typing import Dict, List, Union
import torch


class Logger:
    def __init__(self, logger_csv_filename: str, metrics: List[str], mode: str) -> None:
        """
        Logger class to log the training and validation metrics to a csv file.
            May append to existing file if headers match; elsewise raises an error.

        Args:
            logger_csv_filename (str): Path to a filename where the csv has to be stored.
            metrics (Dict[str, float]): The metrics to be logged.
        """
        self.filename = logger_csv_filename
        mode = mode.lower()
        self.mode = mode.lower()

        new_header = ["epoch_no", f"{mode}_loss"] + [
            f"{mode}_{metric}" for metric in metrics
        ]

        # TODO: do we really need to support appending to existing files?
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                existing_header = f.readline().strip().split(",")
            if set(existing_header) != set(new_header):
                raise ValueError(
                    f"Logger file {self.filename} error: existing header does not match new header."
                    f" Existing header: {existing_header}. New header: {new_header}"
                )
            self.ordered_header = existing_header
        else:
            with open(self.filename, "w") as f:
                f.write(",".join(new_header) + "\n")
            self.ordered_header = new_header

    def write(
        self,
        epoch_number: int,
        loss: Union[float, torch.Tensor],
        epoch_metrics: Dict[str, Union[float, torch.Tensor]],
    ) -> None:
        """
        Write the epoch number, loss and metrics to the csv file.

        Args:
            epoch_number (int): The epoch number.
            loss (float): The loss value.
            epoch_metrics (Dict[str, float]): The metrics to be logged.
        """

        if torch.is_tensor(loss):
            loss = loss.cpu().item()

        row = {"epoch_no": epoch_number, f"{self.mode}_loss": loss}

        for metric, metric_val in epoch_metrics.items():
            if torch.is_tensor(metric_val):
                metric_val = metric_val.cpu().item()
            row[f"{self.mode}_{metric}"] = metric_val

        with open(self.filename, "a") as f:
            line = [row.get(col, "") for col in self.ordered_header]
            line = [str(x) for x in line]
            f.write(",".join(line) + "\n")
