import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from . import networks
import sys

def get_final_layer(final_convolution_layer):
    none_list = [
        "none",
        None,
        "None",
        "regression",
        "classification_but_not_softmax",
        "logits",
        "classification_without_softmax",
    ]

    if final_convolution_layer == "sigmoid":
        final_convolution_layer = torch.sigmoid

    elif final_convolution_layer == "softmax":
        final_convolution_layer = F.softmax

    elif final_convolution_layer in none_list:
        final_convolution_layer = None

    return final_convolution_layer


def get_norm_type(norm_type, dimensions):
    if dimensions == 3:
        if norm_type == "batch":
            norm_type = nn.BatchNorm3d
        elif norm_type == "instance":
            norm_type = nn.InstanceNorm3d
    elif dimensions == 2:
        if norm_type == "batch":
            norm_type = nn.BatchNorm2d
        elif norm_type == "instance":
            norm_type = nn.InstanceNorm2d

    return norm_type



class ModelBase(nn.Module):
    

    def __init__(self, parameters):
        """
        This defines all defaults that the model base uses

        Args:
            parameters (dict): This is a dictionary of all parameters that are needed for the model.
        """
        super(ModelBase, self).__init__()
        
        
        #if not ("gan_mode" in params["model"]):
         #   sys.exit("The 'model' parameter needs 'gan_mode' key to be defined")
        
        if not ("architecture_gen" in parameters["model"]):
            sys.exit("The 'model' parameter needs 'architecture_gen' key to be defined")
        
        if not ("architecture_disc" in parameters["model"]):
            sys.exit("The 'model' parameter needs 'architecture_disc' key to be defined")
            
        self.loss_mode = parameters["model"]["loss_mode"]
        #gan mode will be added to parameter parser
        self.gen_model_name = parameters["model"]["architecture_gen"]
        self.disc_model_name = parameters["model"]["architecture_disc"]
        #These 2 will be added to parser
        
        #self.lr = parameters["learning_rate"]
        self.n_dimensions = parameters["model"]["dimension"]
        self.n_channels = parameters["model"]["num_channels"]
        self.n_classes = parameters["model"]["num_classes"]
        self.base_filters = parameters["model"]["base_filters"]
        self.norm_type = parameters["model"]["norm_type"]
        self.patch_size = parameters["patch_size"]
        self.batch_size = parameters["batch_size"]
        self.amp = parameters["model"]["amp"]
        self.dev = parameters["device"]
        
        # amp is not supported for sdnet
        parameters["model"]["amp"] = False

        
        self.amp, self.device, self.gpu_ids= networks.device_parser(self.amp, self.dev)
        
    # get device name: CPU or GPU

 
        if self.n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            self.Norm = get_norm_type(self.norm_type.lower(), self.n_dimensions)
            self.ReflectionPad = nn.ReflectionPad2d

        elif self.n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            self.Norm = get_norm_type(self.norm_type.lower(), self.n_dimensions)
            self.ReflectionPad = nn.ReflectionPad2d


   
    def set_input(self, input):
       # Unpack input data from the dataloader and perform necessary pre-processing steps.
       # Parameters:
        #    input (dict): includes the data itself and its metadata information.
        
        pass
    
    def forward(self):
   
        pass

    def optimize_parameters(self):
        #Calculate losses, gradients, and update network weights; called in every training iteration
        pass

            
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for parameters in net.parameters():
                    parameters.requires_grad = requires_grad


