import os
import torch


class Logger:
    def __init__(self, logger_csv_filename: str, metrics: list) -> None:
        """
        Logger class to log the training and validation metrics to a csv file.

        Args:
            logger_csv_filename (str): Path to a filename where the csv has to be stored.
            metrics (list): The list of metrics to be logged.
        """
        self.filename = logger_csv_filename
        self.metrics = metrics

    def write_header(self, mode="train"):
        self.csv = open(self.filename, "a")
        if os.stat(self.filename).st_size == 0:
            mode_lower = mode.lower()
            row = "epoch_no," + mode_lower + "_loss,"
            row += (
                ",".join([mode_lower + "_" + metric for metric in self.metrics]) + ","
            )
            row = row[:-1]
            row += "\n"
            self.csv.write(row)
        # else:
        #     print("Found a pre-existing file for logging, now appending logs to that file!")
        self.csv.close()

    def write(self, epoch_number: int, loss: float, epoch_metrics: list) -> None:
        """
        Write the epoch number, loss and metrics to the csv file.

        Args:
            epoch_number (int): The epoch number.
            loss (float): The loss value.
            epoch_metrics (list): The list of metrics to be logged.
        """
        self.csv = open(self.filename, "a")
        row = ""
        row += str(epoch_number) + ","
        if torch.is_tensor(loss):
            row += str(loss.cpu().item())
        else:
            row += str(loss)
        row += ","

        for metric in epoch_metrics:
            if torch.is_tensor(epoch_metrics[metric]):
                row += str(epoch_metrics[metric].cpu().item())
            else:
                row += str(epoch_metrics[metric])
            row += ","
        row = row[:-1]
        self.csv.write(row)
        self.csv.write("\n")
        self.csv.close()

    def close(self):
        self.csv.close()
