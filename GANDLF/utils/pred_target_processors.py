import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from GANDLF.utils.tensor import reverse_one_hot, get_linear_interpolation_mode, one_hot

from typing import Tuple


class AbstractPredictionTargetProcessor(ABC):
    def __init__(self, params: dict):
        """
        Interface for classes that perform specific processing on the target and/or prediction tensors.
        Useful for example for metrics or loss calculations, where some architectures require specific
        processing of the target and/or prediction tensors before the metric or loss can be calculated.
        """
        super().__init__()
        self.params = params

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class DeepSupervisionPredictionTargetProcessor(AbstractPredictionTargetProcessor):
    def __init__(self, params: dict):
        """
        Processor for deep supervision architectures.
        """
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        target = one_hot(target, self.params["model"]["class_list"])
        target_resampled = []
        target_prev = target.detach()
        for i, _ in enumerate(prediction):
            if target_prev[0].shape != prediction[i][0].shape:
                expected_shape = reverse_one_hot(
                    prediction[i][0].detach(), self.params["model"]["class_list"]
                ).shape
                target_prev = F.interpolate(
                    target_prev,
                    size=expected_shape,
                    mode=get_linear_interpolation_mode(len(expected_shape)),
                    align_corners=False,
                )
            target_resampled.append(target_prev)
        return prediction, target_resampled


class IdentityPredictionTargetProcessor(AbstractPredictionTargetProcessor):
    def __init__(self, params: dict):
        """
        No-op processor that returns the input target and prediction tensors.
        Used when no processing is needed.
        """
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        return prediction, target


class PredictionTargetProcessorFactory:
    def __init__(self, params: dict):
        self.params = params

    def get_prediction_target_processor(self) -> AbstractPredictionTargetProcessor:
        if "deep" in self.params["model"]["architecture"].lower():
            return DeepSupervisionPredictionTargetProcessor(self.params)
        return IdentityPredictionTargetProcessor(self.params)
