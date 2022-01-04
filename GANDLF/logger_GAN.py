#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:44 2021

@author: siddhesh
"""

import os
import torch


class Logger:
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

    def write_header(self,loss_names, mode="train"):
        self.csv = open(self.filename, "a")
        if os.stat(self.filename).st_size == 0:
            mode_lower = mode.lower()
            row="epoch_no,"
            for i in range(len(loss_names)):
                row += mode_lower + "_" + loss_names[i] + "_loss,"
            if not mode=="train":
                for metric in self.metrics:
                    row += mode_lower + "_" + metric + ","
            row = row[:-1]
            row += "\n"
            self.csv.write(row)
        # else:
        #     print("Found a pre-existing file for logging, now appending logs to that file!")
        self.csv.close()

    def write(self, epoch_number, loss, *argv):
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
        if type(loss)!=list:
            l_rem=[]
            l_rem.append(loss)
            loss=l_rem
        
        self.csv = open(self.filename, "a")
        row = ""
        row += str(epoch_number) + ","
        for i in range(len(loss)):

            if torch.is_tensor(loss[i]):
                row += str(loss[i].cpu().data.item())
            else:
                row += str(loss[i])
            row += ","
        
        for epoch_metrics in argv:
            for metric in epoch_metrics:
                if torch.is_tensor(epoch_metrics[metric]):
                    row += str(epoch_metrics[metric].cpu().data.item())
                else:
                    row += str(epoch_metrics[metric])
                row += ","
            row = row[:-1]
        self.csv.write(row)
        self.csv.write("\n")
        self.csv.close()

    def close(self):
        self.csv.close()
