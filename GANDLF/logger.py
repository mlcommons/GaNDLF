#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:44 2021

@author: siddhesh
"""

import os
import torch

class Logger():
    def __init__(self, logger_csv_filename, metrics):
        """

        Parameters
        ----------
        logger_csv_filename : String
            Path to a filename where the csv has to be stored
        metric : list
            Should be a list of the metrics

        Returns
        -------
        None.

        """
        self.filename = logger_csv_filename
        self.metrics = metrics

    def write_header(self, mode='train'):
        self.csv = open(self.filename, "a")
        if os.stat(self.filename).st_size == 0:
            if mode.lower() == 'train':
                self.csv.write("epoch_no,train_loss,")
                for metric in self.metrics:
                    self.csv.write("train_"+metric+",")
                    self.csv.write("\b\n")
            else:
                self.csv.write("epoch_no,valid_loss,")
                for metric in self.metrics:
                    self.csv.write("valid_"+metric+",")
                    self.csv.write("\b\n")
        else:
            print("Found a pre-existing file for logging, now appending logs to that file!")
        self.csv.close()

    def write(self, epoch_number, loss, epoch_metrics):
        """

        Parameters
        ----------
        epoch_number : TYPE
            DESCRIPTION.
        loss : TYPE
            DESCRIPTION.
        metrics : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.csv = open(self.filename, "a")
        self.csv.write(str(epoch_number)+","+str(loss)+",")
        for metric in epoch_metrics:
            if torch.is_tensor(epoch_metrics[metric]):
                to_write = str(epoch_metrics[metric].cpu().data.item())
            else:
                to_write = str(epoch_metrics[metric])
            self.csv.write(to_write+",")
        self.csv.write("\b\n")
        self.csv.close()

    def close(self):
        self.csv.close()
