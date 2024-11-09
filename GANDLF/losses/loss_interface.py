import torch
from torch import nn
from abc import ABC, abstractmethod


class AbstractLossFunction(ABC, nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


class WeightedCE(AbstractLossFunction):
    def __init__(self, params: dict):
        """
        Cross entropy loss using class weights if provided.
        """
        super().__init__(params)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) > 1 and target.shape[-1] == 1:
            target = torch.squeeze(target, -1)

        weights = None
        if self.params.get("penalty_weights") is not None:
            num_classes = len(self.params["penalty_weights"])
            assert (
                prediction.shape[-1] == num_classes
            ), f"Number of classes {num_classes} does not match prediction shape {prediction.shape[-1]}"

            weights = torch.tensor(
                list(self.params["penalty_weights"].values()),
                dtype=torch.float32,
                device=target.device,
            )

        cel = nn.CrossEntropyLoss(weight=weights)
        return cel(prediction, target)
