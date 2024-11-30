import lightning.pytorch as pl
from GANDLF.models import get_model
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.metrics.metric_calculators import MetricCalculatorFactory
from GANDLF.utils.pred_target_processors import PredictionTargetProcessorFactory

from copy import deepcopy


class GandlfLightningModule(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.params = deepcopy(params)
        self._initialize_model()
        self._initialize_loss()
        self._initialize_metric_calculators()
        self._initialize_preds_target_processor()

    def _initialize_model(self):
        self.model = get_model(self.params)

    def _initialize_loss(self):
        self.loss = LossCalculatorFactory(self.params).get_loss_calculator()

    def _initialize_metric_calculators(self):
        # TODO can we have situation that metrics are empty?
        self.metric_calculators = MetricCalculatorFactory(
            self.params
        ).get_metric_calculator()

    def _initialize_preds_target_processor(self):
        self.pred_target_processor = PredictionTargetProcessorFactory(
            self.params
        ).get_prediction_target_processor()

    def configure_optimizers(self):
        params = deepcopy(self.params)
        params["model_parameters"] = self.model.parameters()
        optimizer = get_optimizer(params)
        if "scheduler" in self.params:
            params["optimizer_object"] = optimizer
            scheduler = get_scheduler(params)
            return [optimizer], [scheduler]
        return optimizer
