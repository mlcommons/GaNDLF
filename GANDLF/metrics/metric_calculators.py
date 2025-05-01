import torch
from copy import deepcopy
from GANDLF.metrics import get_metrics
from abc import ABC, abstractmethod
from typing import Union


class AbstractMetricCalculator(ABC):
    def __init__(self, params: dict):
        super().__init__()
        self.params = deepcopy(params)
        self._initialize_metrics_dict()

    def _initialize_metrics_dict(self):
        self.metrics_calculators = get_metrics(self.params)

    def _process_metric_value(self, metric_value: Union[torch.Tensor, float]):
        if isinstance(metric_value, float):
            return metric_value
        if metric_value.dim() == 0:
            return metric_value.item()
        else:
            return metric_value.tolist()

    @staticmethod
    def _inject_kwargs_into_params(params, **kwargs):
        for key, value in kwargs.items():
            params[key] = value
        return params

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        pass


class MetricCalculatorSDNet(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs):
        params = deepcopy(self.params)
        params = self._inject_kwargs_into_params(params, **kwargs)

        metric_results = {}

        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_value = (
                metric_calculator(prediction[0], target.squeeze(-1), params)
                .detach()
                .cpu()
            )
            metric_results[metric_name] = self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorDeepSupervision(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs):
        params = deepcopy(self.params)
        params = self._inject_kwargs_into_params(params, **kwargs)
        metric_results = {}

        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_results[metric_name] = 0.0
            for i, _ in enumerate(prediction):
                metric_value = (
                    metric_calculator(prediction[i], target[i], params).detach().cpu()
                )
                metric_results[metric_name] += self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorSimple(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs):
        params = deepcopy(self.params)
        params = self._inject_kwargs_into_params(params, **kwargs)
        metric_results = {}

        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_value = metric_calculator(prediction, target, params).detach().cpu()
            metric_results[metric_name] = self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorFactory:
    def __init__(self, params: dict):
        self.params = params

    def get_metric_calculator(self) -> AbstractMetricCalculator:
        if self.params["model"]["architecture"] == "sdnet":
            return MetricCalculatorSDNet(self.params)
        elif "deep" in self.params["model"]["architecture"].lower():
            return MetricCalculatorDeepSupervision(self.params)
        else:
            return MetricCalculatorSimple(self.params)
