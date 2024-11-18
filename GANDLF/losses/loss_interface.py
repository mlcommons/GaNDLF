import torch
from torch import nn
from abc import ABC, abstractmethod


class AbstractLossFunction(nn.Module, ABC):
    def __init__(self, params: dict):
        nn.Module.__init__(self)
        self.params = params
        self.num_classes = len(params["model"]["class_list"])
        self._initialize_penalty_weights()

    def _initialize_penalty_weights(self):
        default_penalty_weights = torch.ones(self.num_classes)
        self.penalty_weights = self.params.get(
            "penalty_weights", default_penalty_weights
        )

    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


class AbstractSegmentationMultiClassLoss(AbstractLossFunction):
    """
    Base class for loss funcions that are used for multi-class segmentation tasks.
    """

    def __init__(self, params: dict):
        super().__init__(params)

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

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        accumulated_loss = torch.tensor(0.0, device=prediction.device)

        for class_idx in range(self.num_classes):
            current_loss = self._compute_single_class_loss(
                prediction, target, class_idx
            )
            accumulated_loss += (
                self._optional_loss_operations(current_loss)
                * self.penalty_weights[class_idx]
            )

        # TODO shouldn't we always divide by the number of classes?
        if self.penalty_weights is None:
            accumulated_loss /= self.num_classes

        return accumulated_loss


class AbstractRegressionLoss(AbstractLossFunction):
    """
    Base class for loss functions that are used for regression and classification tasks.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.loss_calculator = self._initialize_loss_function_object()
        self.reduction_method = self._initialize_reduction_method()

    def _initialize_reduction_method(self) -> str:
        """
        Initialize the reduction method for the loss function. Defaults to 'mean'.
        """
        loss_params = self.params["loss_function"]
        reduction_method = "mean"
        if isinstance(loss_params, dict):
            reduction_method = loss_params.get("reduction", reduction_method)
            assert reduction_method in [
                "mean",
                "sum",
            ], f"Invalid reduction method defined for loss function: {reduction_method}. Valid options are ['mean', 'sum']"
        return reduction_method

    def _calculate_loss_for_single_class(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for a single class. To be implemented by child classes.
        """
        return self.loss_calculator(prediction, target)

    @abstractmethod
    def _initialize_loss_function_object(self) -> nn.modules.loss._Loss:
        """
        Initialize the loss function object used in the forward method. Has to return
        callable pytorch loss function object.
        """
        pass

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        accumulated_loss = torch.tensor(0.0, device=prediction.device)
        for class_idx in range(self.num_classes):
            accumulated_loss += (
                self._calculate_loss_for_single_class(
                    prediction[:, class_idx, ...], target[:, class_idx, ...]
                )
                * self.penalty_weights[class_idx]
            )

        # TODO I Believe this is how it should be, also for segmentation - take average from all classes, despite weights being present or no
        accumulated_loss /= self.num_classes

        return accumulated_loss
