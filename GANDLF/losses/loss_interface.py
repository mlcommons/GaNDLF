import torch
from torch import nn
from abc import ABC, abstractmethod


class AbstractLossFunction(nn.Module, ABC):
    def __init__(self, params: dict):
        nn.Module.__init__(self)
        self.params = params

    @abstractmethod
    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        pass


class AbstractSegmentationMultiClassLoss(AbstractLossFunction):
    """
    Base class for loss funcions that are used for multi-class segmentation tasks.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.num_classes = len(params["model"]["class_list"])
        self.penalty_weights = params["penalty_weights"]

    def _compute_single_class_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, class_idx: int
    ) -> torch.Tensor:
        """Compute loss for a single class."""
        loss_value = self._single_class_loss_calculator(
            prediction[:, class_idx, ...], target[:, class_idx, ...]
        )
        return 1 - loss_value

    def _optional_loss_operations(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Perform addtional operations of the loss value. Defaults to identity operation.
        If needed, child classes can override this method. Useful in the cases where
        for example, the loss value needs to log-transformed or clipped.
        """
        return loss

    @abstractmethod
    def _single_class_loss_calculator(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a pair of prediction and target tensors. To be implemented by child classes."""
        pass

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        accumulated_loss = torch.tensor(0.0, device=prediction.device)

        for class_idx in range(self.num_classes):
            current_loss = self._compute_single_class_loss(
                prediction, target, class_idx
            )
            current_loss = self._optional_loss_operations(current_loss)

            if self.penalty_weights is not None:
                current_loss = current_loss * self.penalty_weights[class_idx]
            accumulated_loss += current_loss

        if self.penalty_weights is None:
            accumulated_loss /= self.num_classes

        return accumulated_loss
