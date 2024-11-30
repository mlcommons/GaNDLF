import torch
import torchio
import warnings
from copy import deepcopy
import lightning.pytorch as pl
from GANDLF.models import get_model
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.metrics.metric_calculators import MetricCalculatorFactory
from GANDLF.utils.pred_target_processors import PredictionTargetProcessorFactory

from GANDLF.utils import write_training_patches, one_hot

from typing import Tuple, Union, Dict, List


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

    def _handle_dynamic_batch_size_in_differential_privacy_mode(self, subject):
        raise NotImplementedError(
            "Differential privacy is not implemented yet in lightning version"
        )

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        attention_map = None
        if "medcam_enabled" in self.params and self.params["medcam_enabled"]:
            output, attention_map = self.model(images)
            if self.params["model"]["dimension"] == 2:
                attention_map = torch.unsqueeze(attention_map, -1)
        else:
            output = self.model(images)

        return output, attention_map

    def on_train_start(self):
        self.training_metrics: List[Dict[str, float]] = []

    def on_train_end(self):
        self.training_metrics = [
            {
                key: sum([metric[key] for metric in self.training_metrics])
                / len(self.training_metrics)
            }
            for key in self.training_metrics[0]
        ]
        self.log_dict(self.training_metrics, on_epoch=True, prog_bar=True)

        # TODO
        self._print_and_format_metrics()

        self.training_metrics = []

    def training_step(self, subject, batch_idx):
        if self.params["save_training"]:
            write_training_patches(subject, self.params)

        if self.params.get("differential_privacy"):
            self._handle_dynamic_batch_size_in_differential_privacy_mode(subject)

        images = self._prepare_input_batch_from_subject_data(subject)
        labels = self._prepare_label_batch_from_subject_data(subject)
        self._set_spacing_params_for_subject(subject)
        images, labels = self._process_inputs(images, labels)

        model_output, _ = self.forward(images)
        model_output, labels = self.pred_target_processor(model_output, labels)
        loss = self.loss(model_output, labels)
        metric_results = self.metric_calculators(model_output, labels)

        self.training_metrics.append(metric_results)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def _prepare_input_batch_from_subject_data(self, subject):
        image = (  # 5D tensor: (B, C, H, W, D)
            torch.cat(
                [subject[key][torchio.DATA] for key in self.params["channel_keys"]],
                dim=1,
            )
            .float()
            .to(self.device)
        )
        return image

    def _prepare_label_batch_from_subject_data(self, subject):
        if "value_keys" in self.params:
            # classification / regression (when label is scalar) or multilabel classif/regression
            label = torch.cat(
                [subject[key] for key in self.params["value_keys"]], dim=0
            )
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(self.params["batch_size"], len(label)),
                len(self.params["value_keys"]),
            )
        else:
            label = subject["label"][
                torchio.DATA
            ]  # segmentation; label is (B, C, H, W, D) image
        return label.to(self.device)

    def _set_spacing_params_for_subject(self, subject):
        if "spacing" in subject:
            self.params["subject_spacing"] = subject["spacing"]
        else:
            self.params["subject_spacing"] = None

    def _process_inputs(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modify the input images and labels as needed for forward pass, loss
        and metric calculations.
        """

        if labels is not None:
            if self.params["problem_type"] == "segmentation":
                if labels.shape[1] == 3:
                    labels = labels[:, 0, ...].unsqueeze(1)
                    warnings.warn(
                        "The label image is an RGB image, only the first channel will be used."
                    )

            assert len(labels) == len(images)

        # for segmentation remove the depth dimension from the label.
        # for classification / regression, flattens class / reg label from list (possible in multilabel) to scalar
        # TODO: second condition is crutch - in some cases label is passed as 1-d Tensor (B,) and if Batch size is 1,
        #  it is squeezed to scalar tensor (0-d) and the future logic fails
        if labels is not None and len(labels.shape) != 1:
            labels = labels.squeeze(-1)

        if self.params["problem_type"] == "segmentation":
            labels = one_hot(labels, self.params["model"]["class_list"])

        if self.params["model"]["dimension"] == 2:
            images = images.squeeze(
                -1
            )  # removing depth, as torchio adds last dimension for 2D images

        return images, labels

    def _print_and_format_metrics(self):
        pass
