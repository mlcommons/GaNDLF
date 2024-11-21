import sys
import torch
from .loss_interface import AbstractSegmentationLoss, AbstractLossFunction


class MulticlassDiceLoss(AbstractSegmentationLoss):
    """
    This class computes the Dice loss between two tensors.
    """

    def _single_class_loss_calculator(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice score for a single class.

        Args:
            prediction (torch.Tensor): Network's predicted segmentation mask
            target (torch.Tensor): Target segmentation mask

        Returns:
            torch.Tensor: The computed dice score.
        """
        predicted_flat = prediction.flatten()
        label_flat = target.flatten()
        intersection = (predicted_flat * label_flat).sum()

        dice_score = (2.0 * intersection + sys.float_info.min) / (
            predicted_flat.sum() + label_flat.sum() + sys.float_info.min
        )

        return dice_score


class MulticlassDiceLogLoss(MulticlassDiceLoss):
    def _optional_loss_operations(self, loss):
        return -torch.log(
            loss + torch.finfo(torch.float32).eps
        )  # epsilon for numerical stability


class MulticlassMCCLoss(AbstractSegmentationLoss):
    """
    This class computes the Matthews Correlation Coefficient (MCC) loss between two tensors.
    """

    def _single_class_loss_calculator(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MCC score for a single class.

        Args:
            prediction (torch.Tensor): Network's predicted segmentation mask
            target (torch.Tensor): Target segmentation mask

        Returns:
            torch.Tensor: The computed MCC score.
        """
        tp = torch.sum(torch.mul(prediction, target))
        tn = torch.sum(torch.mul((1 - prediction), (1 - target)))
        fp = torch.sum(torch.mul(prediction, (1 - target)))
        fn = torch.sum(torch.mul((1 - prediction), target))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        # Adding epsilon to the denominator to avoid divide-by-zero errors.
        denominator = (
            torch.sqrt(
                torch.add(tp, 1, fp)
                * torch.add(tp, 1, fn)
                * torch.add(tn, 1, fp)
                * torch.add(tn, 1, fn)
            )
            + torch.finfo(torch.float32).eps
        )

        return torch.div(numerator.sum(), denominator.sum())


class MulticlassMCLLogLoss(MulticlassMCCLoss):
    def _optional_loss_operations(self, loss):
        return -torch.log(
            loss + torch.finfo(torch.float32).eps
        )  # epsilon for numerical stability


class MulticlassTverskyLoss(AbstractSegmentationLoss):
    """
    This class computes the Tversky loss between two tensors.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        loss_params = params["loss_function"]
        self.alpha = 0.5
        self.beta = 0.5
        if isinstance(loss_params, dict):
            self.alpha = loss_params.get("alpha", self.alpha)
            self.beta = loss_params.get("beta", self.beta)

    def _single_class_loss_calculator(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky score for a single class.

        Args:
            prediction (torch.Tensor): Network's predicted segmentation mask
            target (torch.Tensor): Target segmentation mask

        Returns:
            torch.Tensor: The computed Tversky score.
        """
        predicted_flat = prediction.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        true_positives = (predicted_flat * target_flat).sum()
        false_positives = ((1 - target_flat) * predicted_flat).sum()
        false_negatives = (target_flat * (1 - predicted_flat)).sum()

        numerator = true_positives
        denominator = (
            true_positives + self.alpha * false_positives + self.beta * false_negatives
        )
        loss = (numerator + sys.float_info.min) / (denominator + sys.float_info.min)

        return loss


class MulticlassFocalLoss(AbstractSegmentationLoss):
    """
    This class computes the Focal loss between two tensors.
    """

    def __init__(self, params: dict):
        super().__init__(params)

        self.ce_loss_helper = torch.nn.CrossEntropyLoss(reduction="none")
        loss_params = params["loss_function"]
        self.alpha = 1.0
        self.gamma = 2.0
        self.output_aggregation = "sum"
        if isinstance(loss_params, dict):
            self.alpha = loss_params.get("alpha", self.alpha)
            self.gamma = loss_params.get("gamma", self.gamma)
            self.output_aggregation = loss_params.get(
                "size_average",
                self.output_aggregation,  # naming mismatch of key due to keeping API consistent with config format
            )
        assert self.output_aggregation in [
            "sum",
            "mean",
        ], f"Invalid output aggregation method defined for Foal Loss: {self.output_aggregation}. Valid options are ['sum', 'mean']"

    def _single_class_loss_calculator(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for a single class. It is based on the following formulas:
            FocalLoss(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            CrossEntropy(pred, target) = -log(pred) if target = 1 else -log(1 - pred)
            CrossEntropy(p_t) = CrossEntropy(pred, target) = -log(p_t)
            p_t = p if target = 1 else 1 - p
        """
        ce_loss = self.ce_loss_helper(prediction, target)
        p_t = torch.exp(-ce_loss)
        loss = -self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.sum() if self.output_aggregation == "sum" else loss.mean()

    def _compute_single_class_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, class_idx: int
    ) -> torch.Tensor:
        """Compute loss for a single class."""
        loss_value = self._single_class_loss_calculator(
            prediction[:, class_idx, ...], target[:, class_idx, ...]
        )
        return loss_value  # no need to subtract from 1 in this case, hence the override


class KullbackLeiblerDivergence(AbstractLossFunction):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Kullback-Leibler divergence between two Gaussian distributions.

        Args:
            mu (torch.Tensor): The mean of the first Gaussian distribution.
            logvar (torch.Tensor): The logarithm of the variance of the first Gaussian distribution.

        Returns:
            torch.Tensor: The computed Kullback-Leibler divergence
        """
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return loss.mean()
