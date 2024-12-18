import torch
from torch import nn
from .loss_interface import AbstractRegressionLoss


class CrossEntropyLoss(AbstractRegressionLoss):
    """
    This class computes the cross entropy loss between two tensors.
    """

    def _initialize_loss_function_object(self):
        return nn.CrossEntropyLoss(reduction=self.reduction_method)


class BinaryCrossEntropyLoss(AbstractRegressionLoss):
    """
    This class computes the binary cross entropy loss between two tensors.
    """

    def _initialize_loss_function_object(self):
        return nn.BCELoss(reduction=self.reduction_method)


class BinaryCrossEntropyWithLogitsLoss(AbstractRegressionLoss):
    """
    This class computes the binary cross entropy loss with logits between two tensors.
    """

    def _initialize_loss_function_object(self):
        return nn.BCEWithLogitsLoss(reduction=self.reduction_method)


class BaseLossWithScaledTarget(AbstractRegressionLoss):
    """
    General interface for the loss functions requiring scaling of the target tensor.
    """

    def _initialize_scaling_factor(self):
        loss_params: dict = self.params["loss_function"]
        self.scaling_factor = loss_params.get("scaling_factor", 1.0)
        if isinstance(loss_params, dict):
            self.scaling_factor = loss_params.get("scaling_factor", self.scaling_factor)
        return self.scaling_factor

    def _calculate_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.loss_calculator(prediction, target * self.scaling_factor)


class L1Loss(BaseLossWithScaledTarget):
    """
    This class computes the L1 loss between two tensors.
    """

    def _initialize_loss_function_object(self):
        return nn.L1Loss(reduction=self.reduction_method)


class MSELoss(BaseLossWithScaledTarget):
    """
    This class computes the mean squared error loss between two tensors.
    """

    def _initialize_loss_function_object(self):
        return nn.MSELoss(reduction=self.reduction_method)
