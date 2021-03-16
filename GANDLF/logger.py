#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:44 2021

@author: siddhesh
"""

import os

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
        self.csv = open(self.filename, "a")

    def write_header(self, mode='train'):
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
        self.csv.write(str(epoch_number)+","+str(loss)+",")
        for metric in metrics:
            self.csv.write(epoch_metrics[metric]+",")
        self.csv.write("\b\n")

    def close(self):
        self.csv.close()
