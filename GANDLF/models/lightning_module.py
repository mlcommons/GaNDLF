import os
import sys
import time
import psutil
import torch
import torchio
import warnings
import numpy as np
import SimpleITK as sitk
from medcam import medcam
from copy import deepcopy
from statistics import mean
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from GANDLF.logger import Logger
from GANDLF.models import get_model
from GANDLF.metrics import overall_stats
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.data.post_process import global_postprocessing_dict
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.metrics.metric_calculators import MetricCalculatorFactory
from GANDLF.utils.pred_target_processors import PredictionTargetProcessorFactory

from GANDLF.utils import (
    optimize_and_save_model,
    write_training_patches,
    one_hot,
    reverse_one_hot,
    print_model_summary,
    get_date_time,
    save_model,
    load_model,
    version_check,
    get_filename_extension_sanitized,
    resample_image,
    BEST_MODEL_PATH_END,
    INITIAL_MODEL_PATH_END,
    LATEST_MODEL_PATH_END,
)

from overrides import override
from typing import Tuple, Union, Dict, List, Any


class GandlfLightningModule(pl.LightningModule):
    CLASSIFICATION_REGRESSION_RESULTS_HEADER = "Epoch,SubjectID,PredictedValue\n"

    def __init__(self, params: dict, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.params = deepcopy(params)
        self.current_best_loss = sys.float_info.max
        self.wait_count_before_early_stopping = 0
        self._problem_type_is_regression = params["problem_type"] == "regression"
        self._problem_type_is_classification = (
            params["problem_type"] == "classification"
        )
        self._problem_type_is_segmentation = params["problem_type"] == "segmentation"
        self._initialize_model()
        self._initialize_loss()
        self._initialize_metric_calculators()
        self._initialize_preds_target_processor()
        self._initialize_model_save_paths()

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

    def _initialize_model_save_paths(self):
        self.model_paths = {
            "best": os.path.join(
                self.output_dir,
                self.params["model"]["architecture"] + BEST_MODEL_PATH_END,
            ),
            "initial": os.path.join(
                self.output_dir,
                self.params["model"]["architecture"] + INITIAL_MODEL_PATH_END,
            ),
            "latest": os.path.join(
                self.output_dir,
                self.params["model"]["architecture"] + LATEST_MODEL_PATH_END,
            ),
        }

    @rank_zero_only
    def _save_model(self, epoch, save_path, onnx_export):
        save_model(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizers().optimizer.state_dict(),
                "loss": self.current_best_loss,
            },
            model=self.model,
            params=self.params,
            path=save_path,
            onnx_export=onnx_export,
        )

    @staticmethod
    def _ensure_proper_type_of_metric_values_for_progbar(
        metric_results_dict: Dict[str, Any]
    ) -> Dict[str, float]:
        parsed_results_dict = deepcopy(metric_results_dict)
        for metric_name, metric_value in metric_results_dict.items():
            if isinstance(metric_value, list):
                for n, metric_value_for_given_class in enumerate(metric_value):
                    parsed_results_dict[
                        metric_name + f"_class_{n}"
                    ] = metric_value_for_given_class
                del parsed_results_dict[metric_name]
        return parsed_results_dict

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
        self._print_initialization_info()
        self._set_training_start_time()
        self._print_channels_info()
        self._try_to_load_previous_best_model()
        self._try_to_save_initial_model()
        self._initialize_train_logger()
        self._initialize_training_epoch_containers()

        if "medcam" in self.params:
            self._inject_medcam_module()
            self.params["medcam_enabled"] = False  # Medcam
        if "differential_privacy" in self.params:
            self._initialize_differential_privacy()

    def _try_to_load_previous_best_model(self):
        if os.path.exists(self.model_paths["best"]):
            try:
                checkpoint_dict = load_model(self.model_paths["best"], self.device)
                version_check(
                    self.params["version"], version_to_check=checkpoint_dict["version"]
                )
                # I am purposefully omitting the line below, as "previous_parameters" are not used anywhere
                # params["previous_parameters"] = main_dict.get("parameters", None)

                self.model.load_state_dict(checkpoint_dict["model_state_dict"])
                self.optimizers().optimizer.load_state_dict(
                    checkpoint_dict["optimizer_state_dict"]
                )
                self.current_epoch = checkpoint_dict["epoch"]
                self.trainer.callback_metrics["val_loss"] = checkpoint_dict["loss"]
            except Exception as e:
                warnings.warn(
                    f"Previous best model found under path {self.model_paths['best']}, but error occurred during loading: {e}; Continuing training with new model"
                )
        else:
            warnings.warn(
                f"No previous best model found under the path {self.model_paths['best']}; Training from scratch"
            )

    def _try_to_save_initial_model(self):
        if not os.path.exists(self.model_paths["initial"]):
            self._save_model(self.current_epoch, self.model_paths["initial"], False)
            print(f"Initial model saved at {self.model_paths['initial']}")
        else:
            print(
                f"Initial model already exists at {self.model_paths['initial']}; Skipping saving"
            )

    def _inject_medcam_module(self):
        self.model = medcam.inject(
            self.model,
            output_dir=os.path.join(
                self.output_dir, "attention_maps", self.params["medcam"]["backend"]
            ),
            backend=self.params["medcam"]["backend"],
            layer=self.params["medcam"]["layer"],
            save_maps=False,
            return_attention=True,
            enabled=False,
        )

    def _initialize_train_logger(self):
        self.train_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_training.csv"),
            metrics=list(self.params["metrics"]),
            mode="train",
        )

    @rank_zero_only
    def _set_training_start_time(self):
        self.training_start_time = time.time()

    @rank_zero_only
    def _print_initialization_info(self):
        if not (os.environ.get("HOSTNAME") is None):
            print("Hostname :", os.environ.get("HOSTNAME"), flush=True)
        if self.params["verbose"]:
            print("Initializing training at :", get_date_time(), flush=True)
        if self.params["model"]["print_summary"]:
            self._print_model_summary()

    def _print_model_summary(self):
        print_model_summary(
            self.model,
            self.params["batch_size"],
            self.params["model"]["num_channels"],
            self.params["patch_size"],
        )

    def _initialize_training_epoch_containers(self):
        self.train_losses: List[torch.Tensor] = []
        self.training_metric_values: List[Dict[str, float]] = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_predictions: List[torch.Tensor] = []
            self.train_labels: List[torch.Tensor] = []

    @rank_zero_only
    def _print_channels_info(self):
        print("Number of channels : ", self.params["model"]["num_channels"])

    def training_step(self, subject, batch_idx):
        if self.params.get("save_training"):
            write_training_patches(subject, self.params)

        if self.params.get("differential_privacy"):
            self._handle_dynamic_batch_size_in_differential_privacy_mode(subject)

        images = self._prepare_images_batch_from_subject_data(subject)
        labels = self._prepare_labels_batch_from_subject_data(subject)
        self._set_spacing_params_for_subject(subject)
        images = self._process_images(images)
        labels = self._process_labels(labels)
        model_output, _ = self.forward(images)
        model_output, labels = self.pred_target_processor(model_output, labels)

        loss = self.loss(model_output, labels)
        metric_results = self.metric_calculators(model_output, labels)
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_labels.append(labels.detach().cpu())
            self.train_predictions.append(
                torch.argmax(model_output, dim=1).detach().cpu()
            )

        self.train_losses.append(loss.detach().cpu())
        self.training_metric_values.append(metric_results)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            self._ensure_proper_type_of_metric_values_for_progbar(metric_results),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _prepare_images_batch_from_subject_data(self, subject_data: torchio.Subject):
        """
        Concatenates the images from the subject data into a single tensor.

        Args:
            subject_data (torchio.Subject): The torchio.Subject object containing the images.
        Can be also a set of already extracted patches.

        Returns:
            images_batch (torch.Tensor): The concatenated images from the subject data
        of shape (B, C, H, W, D).

        """
        images_batch = torch.cat(
            [subject_data[key][torchio.DATA] for key in self.params["channel_keys"]],
            dim=1,
        )
        return images_batch

    def _prepare_labels_batch_from_subject_data(self, subject: torchio.Subject):
        if self._problem_type_is_regression or self._problem_type_is_classification:
            label = torch.cat(
                [subject[key] for key in self.params["value_keys"]], dim=0
            )
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(self.params["batch_size"], len(label)),
                len(self.params["value_keys"]),
            )
        else:
            # segmentation; label is (B, C, H, W, D) image
            label = subject["label"][torchio.DATA]

        return label

    def _set_spacing_params_for_subject(self, subject):
        if "spacing" in subject:
            self.params["subject_spacing"] = subject["spacing"]
        else:
            self.params["subject_spacing"] = None

    # TODO this shoudl be separate for
    def _process_images(self, images: torch.Tensor):
        """
        Modify the input images and labels as needed for forward pass, loss
        and metric calculations.
        """

        if self.params["model"]["dimension"] == 2:
            # removing depth, as torchio adds last dimension for 2D images
            images = images.squeeze(-1)

        return images

    def _process_labels(self, labels: torch.Tensor):
        """
        Modify the input labels as needed for forward pass, loss
        and metric calculations.
        """

        if self._problem_type_is_segmentation:
            if labels.shape[1] == 3:
                labels = labels[:, 0, ...].unsqueeze(1)
                warnings.warn(
                    "The label image is an RGB image, only the first channel will be used."
                )

        # for segmentation remove the depth dimension from the label.
        # for classification / regression, flattens class / reg label from list (possible in multilabel) to scalar
        # TODO: second condition is crutch - in some cases label is passed as 1-d Tensor (B,) and if Batch size is 1,
        #  it is squeezed to scalar tensor (0-d) and the future logic fails
        if len(labels.shape) != 1:
            labels = labels.squeeze(-1)

        if self._problem_type_is_segmentation:
            labels = one_hot(labels, self.params["model"]["class_list"])

        return labels

    def _handle_dynamic_batch_size_in_differential_privacy_mode(self, subject):
        raise NotImplementedError(
            "Differential privacy is not implemented yet in lightning version"
        )

    def _initialize_differential_privacy(self):
        raise NotImplementedError(
            "Differential privacy is not implemented yet in lightning version"
        )

    def on_train_epoch_start(self):
        self._set_epoch_start_time()
        if self.params["track_memory_usage"]:
            self._write_epoch_start_process_resource_usage(self.current_epoch)
        if self.params["verbose"]:
            self._print_epoch_start_time()

    def _write_epoch_start_process_resource_usage(self, epoch):
        filename = f"memory_usage_local_rank_{self.local_rank}_global_rank_{self.global_rank}.csv"
        memory_stats_dir = self._prepare_memory_stats_save_dir()
        full_filepath = os.path.join(memory_stats_dir, filename)
        file_write_mode = "a" if os.path.exists(full_filepath) else "w"
        using_cuda = "cuda" in self.device.type

        memory_info_string = "Epoch,Memory_Total,Memory_Available,Memory_Percent_Free,Memory_Usage,"  # used to write output
        if using_cuda:
            memory_info_string += (
                "CUDA_active.all.peak,CUDA_active.all.current,CUDA_active.all.allocated"
            )
        memory_info_string += "\n"

        host_memory_stats = psutil.virtual_memory()
        memory_info_string += (
            str(epoch)
            + ","
            + str(host_memory_stats[0])
            + ","
            + str(host_memory_stats[1])
            + ","
            + str(host_memory_stats[2])
            + ","
            + str(host_memory_stats[3])
        )
        if using_cuda:
            cuda_memory_stats = torch.cuda.memory_stats()
            memory_info_string += (
                ","
                + str(cuda_memory_stats["active.all.peak"])
                + ","
                + str(cuda_memory_stats["active.all.current"])
                + ","
                + str(cuda_memory_stats["active.all.allocated"])
            )
        memory_info_string += ",\n"
        with open(full_filepath, file_write_mode) as file_mem:
            file_mem.write(memory_info_string)

    @rank_zero_only
    def _prepare_memory_stats_save_dir(self):
        memory_stats_dir = os.path.join(self.output_dir, "memory_stats")
        os.makedirs(memory_stats_dir, exist_ok=True)
        return memory_stats_dir

    @rank_zero_only
    def _print_epoch_start_time(self):
        print("Epoch start time : ", get_date_time(), flush=True)

    @rank_zero_only
    def _set_epoch_start_time(self):
        self.epoch_start_time = time.time()

    # TODO when used with multiple GPUs, this should produce multiple logs
    # for each GPU. We should think on doing allgather here in a function
    # that is called on the main process (rank 0)

    def on_train_epoch_end(self):
        epoch_metrics = {}
        metric_names = self.training_metric_values[0].keys()
        for metric_name in metric_names:
            metric_values = [x[metric_name] for x in self.training_metric_values]
            epoch_metrics[
                metric_name
            ] = self._compute_metric_mean_across_values_from_batches(metric_values)

        if self._problem_type_is_regression or self._problem_type_is_classification:
            training_epoch_average_metrics_overall = overall_stats(
                torch.cat(self.train_predictions),
                torch.cat(self.train_labels),
                self.params,
            )
            epoch_metrics.update(training_epoch_average_metrics_overall)
        train_losses_gathered = self.all_gather(self.train_losses)
        mean_loss = torch.mean(torch.stack(train_losses_gathered)).item()

        self._clear_training_epoch_containers()

        self.train_logger.write(
            self.current_epoch,
            mean_loss,
            self._ensure_proper_metric_formatting_for_logging(epoch_metrics),
        )
        self.log_dict(
            self._ensure_proper_type_of_metric_values_for_progbar(epoch_metrics),
            on_epoch=True,
            prog_bar=True,
        )

        if self.params["verbose"]:
            self._print_epoch_end_time()
        if self.params["model"]["save_at_every_epoch"]:
            self._save_epoch_end_checkpoint()
        if os.path.exists(self.model_paths["latest"]):
            os.remove(self.model_paths["latest"])
        self._save_model(self.current_epoch, self.model_paths["latest"], False)
        print(f"Latest model saved")

    @staticmethod
    def _compute_metric_mean_across_values_from_batches(
        metric_values: List[Union[float, List[float]]]
    ) -> Union[float, List[float]]:
        """
        Given a list of metrics calculated for each batch, computes the mean across all batches.
        Takes into account case where metric is a list of values (e.g. for each class).
        """
        if isinstance(metric_values[0], list):
            return [
                mean([batch_metrics[i] for batch_metrics in metric_values])
                for i in range(len(metric_values[0]))
            ]
        return mean(metric_values)

    @staticmethod
    def _ensure_proper_metric_formatting_for_logging(metrics_dict: dict) -> dict:
        """
        Helper function to ensure that all metric values are in the correct format for logging.
        """
        output_metrics_dict = deepcopy(metrics_dict)
        for metric in metrics_dict.keys():
            if isinstance(metrics_dict[metric], list):
                output_metrics_dict[metric] = ("_").join(
                    str(metrics_dict[metric])
                    .replace("[", "")
                    .replace("]", "")
                    .replace(" ", "")
                    .split(",")
                )

        return output_metrics_dict

    def _save_epoch_end_checkpoint(self):
        epoch_save_path = os.path.join(
            self.output_dir,
            self.params["model"]["architecture"]
            + "_epoch_"
            + str(self.current_epoch)
            + ".pth.tar",
        )
        self._save_model(self.current_epoch, epoch_save_path, False)
        print(f"Epoch model saved.")

    @rank_zero_only
    def _print_epoch_end_time(self):
        print(
            "Time taken for epoch : ",
            (time.time() - self.epoch_start_time) / 60,
            " mins",
            flush=True,
        )

    def _clear_training_epoch_containers(self):
        self.train_losses = []
        self.training_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_predictions = []
            self.train_labels = []

    def on_train_end(self):
        if os.path.exists(self.model_paths["best"]):
            # Why don't we handle it here with the full save_model function?
            optimize_and_save_model(
                self.model, self.params, self.model_paths["best"], onnx_export=True
            )
        self._print_total_training_time()

    @rank_zero_only
    def _print_total_training_time(self):
        print(
            "Total time taken for training : ",
            (time.time() - self.training_start_time) / 60,
            " mins",
            flush=True,
        )

    def on_validation_start(self):
        self._initialize_validation_epoch_containers()
        self._initialize_validation_logger()

    def _initialize_validation_epoch_containers(self):
        self.val_losses: List[torch.Tensor] = []
        self.validation_metric_values: List[Dict[str, float]] = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.val_predictions: List[float] = []
            self.val_labels: List[float] = []
            if self.params["save_outputs"]:
                self.rows_to_write: List[str] = []

    def _initialize_validation_logger(self):
        self.val_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_validation.csv"),
            metrics=list(self.params["metrics"]),
            mode="val",
            add_epsilon=bool(self.params.get("differential_privacy")),
        )

    def on_validation_epoch_start(self):
        # TODO this is dead code both here and in original loops
        # by default medcam is injected at the training and ["medcam_enabled"] is set to False
        # so this block is never executed
        if self.params["medcam_enabled"]:
            self.model.enable_medcam()
            self.params["medcam_enabled"] = True
        self._current_validation_epoch_save_dir = os.path.join(
            self.output_dir, f"output_validation", f"epoch_{self.current_epoch}"
        )
        self._ensure_path_exists(self._current_validation_epoch_save_dir)

    def validation_step(self, subject, batch_idx):
        if self.params["verbose"]:
            self._print_currently_processed_validation_subject(subject)
        # TODO spacing in global params is going to effectively block any paralllelism, as the
        # spacing is going to unpredicatably change across GPUs
        self.params["subject_spacing"] = subject.get("spacing", None)
        subject_dict = self._initialize_validation_subject_dict(subject)
        if self._problem_type_is_regression or self._problem_type_is_classification:
            model_output = self._regression_or_classification_validation_step(
                subject_dict, subject["subject_id"][0]
            )
            label = self._initialize_validation_label_ground_truth_classification_or_regression(
                subject
            )
        else:
            model_output = self._segmentation_validation_step(subject, subject_dict)
            label = self._initialize_validation_label_ground_truth_segmentation(subject)

        model_output, label = self.pred_target_processor(model_output, label)
        # TODO do something with 4 lines below because I can't look at it
        if label.shape[0] == 3:
            label = label[0, ...].unsqueeze(0)
        label = label.unsqueeze(0)
        label = self._process_labels(label)

        loss = self.loss(model_output, label)
        metric_results = self.metric_calculators(model_output, label)
        self.val_losses.append(loss)
        self.validation_metric_values.append(metric_results)
        if self._problem_type_is_regression or self._problem_type_is_classification:
            model_prediction = (
                torch.argmax(model_output[0], 0)
                if self._problem_type_is_classification
                else model_output[0]
            )  # TODO am I right here? For regression, we should not take argmax
            self.val_predictions.append(model_prediction.item())
            self.val_labels.append(label.item())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            self._ensure_proper_type_of_metric_values_for_progbar(metric_results),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _print_currently_processed_validation_subject(self, subject):
        print("== Current subject:", subject["subject_id"], flush=True)

    def _initialize_validation_subject_dict(self, subject):
        subject_dict = {}
        subject_dict["label"] = torchio.LabelMap(
            path=subject["label"]["path"],
            tensor=subject["label"]["data"].squeeze(0),
            affine=subject["label"]["affine"].squeeze(0),
        )

        if self._problem_type_is_regression or self._problem_type_is_classification:
            for key in self.params["value_keys"]:
                subject_dict["value_" + key] = subject[key]

        for channel_key in self.params["channel_keys"]:
            subject_dict[channel_key] = torchio.ScalarImage(
                path=subject[channel_key]["path"],
                tensor=subject[channel_key]["data"].squeeze(0),
                affine=subject[channel_key]["affine"].squeeze(0),
            )
        return subject_dict

    def _initialize_validation_label_ground_truth_classification_or_regression(
        self, subject
    ):
        return torch.cat([subject[key] for key in self.params["value_keys"]], dim=0)

    def _initialize_validation_label_ground_truth_segmentation(self, subject):
        return subject["label"]["data"].squeeze(0)

    def _regression_or_classification_validation_step(self, subject_dict, subject_id):
        def _prepare_row_for_output_csv(
            subject_id: str, prediction_logit: float, epoch: int
        ):
            return f"{epoch},{subject_id},{prediction_logit}\n"

        def _process_prediction_logit_for_row_writing(
            prediction_logit: torch.Tensor, scaling_factor: float
        ):
            return prediction_logit.cpu().max().item() / scaling_factor

        prediction_logit = self._get_predictions_on_subject_using_label_sampler(
            subject_dict
        )
        if self.params["save_outputs"]:
            processed_logit = _process_prediction_logit_for_row_writing(
                prediction_logit, self.params["scaling_factor"]
            )
            self.rows_to_write.append(
                _prepare_row_for_output_csv(
                    subject_id, processed_logit, self.current_epoch
                )
            )
        return prediction_logit

    # TODO this whole logic can be packed into something separate, as it is only used
    # in validation of regression and classification problems
    def _get_predictions_on_subject_using_label_sampler(self, subject_dict):
        def _prepare_images_batch_from_patch_regression_or_classification_validation_mode(
            patches_batch,
        ):
            """
            Validation mode processing for regression and classification problems requires
            a different approach to preparing the images batch.
            """
            images_batch_from_patches = torch.cat(
                [
                    patches_batch[key][torchio.DATA]
                    for key in self.params["channel_keys"]
                ],
                dim=0,
            ).unsqueeze(0)
            if images_batch_from_patches.shape[-1] == 1:
                images_batch_from_patches = torch.squeeze(images_batch_from_patches, -1)
            return images_batch_from_patches

        sampler = torchio.data.LabelSampler(self.params["patch_size"])
        tio_subject = torchio.Subject(subject_dict)
        patch_loader = sampler(
            tio_subject, num_patches=self.params["q_samples_per_volume"]
        )

        total_logits_for_all_patches = 0.0
        for patches_batch in patch_loader:
            images_from_patches = _prepare_images_batch_from_patch_regression_or_classification_validation_mode(
                patches_batch
            )
            images_from_patches = self._process_images(images_from_patches)
            model_output, _ = self.forward(images_from_patches)
            total_logits_for_all_patches += model_output

        return total_logits_for_all_patches / self.params["q_samples_per_volume"]

    def _segmentation_validation_step(self, subject, subject_dict):
        predicted_segmentation_mask = (
            self._get_predictions_on_subject_using_grid_sampler(subject_dict)
        )
        if self.params["save_outputs"]:
            self._save_predictions_for_segmentation_subject(
                predicted_segmentation_mask, subject
            )
        return predicted_segmentation_mask

    def _get_predictions_on_subject_using_grid_sampler(self, subject_dict):
        def _ensure_output_shape_compatibility_with_torchio(model_output: torch.Tensor):
            # for 2d images where the depth is removed, add it back
            if (
                self.params["model"]["dimension"] == 2
                and self._problem_type_is_segmentation
            ):
                model_output = model_output.unsqueeze(-1)
            return model_output

        grid_sampler, patch_loader = self._prepare_grid_aggregator_and_patch_loader(
            subject_dict
        )
        prediction_aggregator = torchio.inference.GridAggregator(
            grid_sampler,
            overlap_mode=self.params["inference_mechanism"]["grid_aggregator_overlap"],
        )
        if self.params["medcam_enabled"]:
            medcam_attention_map_aggregator = torchio.inference.GridAggregator(
                grid_sampler,
                overlap_mode=self.params["inference_mechanism"][
                    "grid_aggregator_overlap"
                ],
            )
        for patches_batch in patch_loader:
            images_from_patches = self._prepare_images_batch_from_subject_data(
                patches_batch
            )
            images_from_patches = self._process_images(images_from_patches)
            model_output, attention_map = self.forward(images_from_patches)
            model_output = _ensure_output_shape_compatibility_with_torchio(model_output)
            if self.params["medcam_enabled"]:
                medcam_attention_map_aggregator.add_batch(attention_map)
            prediction_aggregator.add_batch(
                model_output, patches_batch[torchio.LOCATION]
            )
        if self.params["medcam_enabled"]:
            attention_map = medcam_attention_map_aggregator.get_output_tensor()
            for i, n in enumerate(attention_map):
                self.model.save_attention_map(
                    n.squeeze(), raw_input=images_from_patches[i].squeeze(-1)
                )
        return prediction_aggregator.get_output_tensor().unsqueeze(0)

    def _prepare_grid_aggregator_and_patch_loader(self, subject_dict):
        grid_sampler = torchio.inference.GridSampler(
            torchio.Subject(subject_dict),
            self.params["patch_size"],
            patch_overlap=self.params["inference_mechanism"]["patch_overlap"],
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

        return grid_sampler, patch_loader

    # TODO will also suffer from the same issue as the training step with
    # synchronization across GPUs
    def on_validation_epoch_end(self):
        validation_epoch_average_metrics = {}
        metric_names = self.validation_metric_values[0].keys()
        for metric_name in metric_names:
            metric_values = [x[metric_name] for x in self.validation_metric_values]
            validation_epoch_average_metrics[
                metric_name
            ] = self._compute_metric_mean_across_values_from_batches(metric_values)

        if self._problem_type_is_regression or self._problem_type_is_classification:
            validation_epoch_average_metrics_overall = overall_stats(
                torch.tensor(self.val_predictions),
                torch.tensor(self.val_labels),
                self.params,
            )
            validation_epoch_average_metrics.update(
                validation_epoch_average_metrics_overall
            )

        val_losses_gathered = self.all_gather(self.val_losses)
        mean_loss = torch.mean(torch.stack(val_losses_gathered)).item()

        self._clear_validation_epoch_containers()

        self.val_logger.write(
            self.current_epoch,
            mean_loss,
            self._ensure_proper_metric_formatting_for_logging(
                validation_epoch_average_metrics
            ),
        )

        self._check_if_early_stopping(mean_loss)
        if self.params["save_outputs"]:
            if self._problem_type_is_regression or self._problem_type_is_classification:
                # Do allgather for predictions and labels
                self._save_predictions_for_regression_or_classification(
                    self.rows_to_write
                )
            # else:
            #     self._save_predictions_for_segmentation()

    def _clear_validation_epoch_containers(self):
        self.val_losses = []
        self.validation_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.val_predictions = []
            self.val_labels = []
            if self.params["save_outputs"]:
                self.rows_to_write = []

    @rank_zero_only
    def _ensure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # TODO called at the validation step, NOT at the end of the epoch - we want to avoid
    # saving all predictions for all subjects for the end of the epoch
    def _save_predictions_for_segmentation_subject(
        self, predicted_segmentation_mask: torch.Tensor, subject: torchio.Subject
    ):
        """
        Saves the predicted segmentation mask for a given subject.

        Args:
            predicted_segmentation_mask: The predicted segmentation mask, extracted
        from the grid aggregator when all validation patches for this subject have
        been processed.
            subject: The subject for which the segmentation mask was predicted, used
        to extract the metadata.
        """

        def _convert_subject_to_sikt_format(subject: torchio.Subject):
            return torchio.ScalarImage(
                tensor=subject["1"]["data"].squeeze(0),
                affine=subject["1"]["affine"].squeeze(0),
            ).as_sitk()

        def _postprocess_raw_segmentation_mask(
            segmentation_mask: np.ndarray, params: dict
        ):
            for postprocessor in params["data_postprocessing"]:
                for _class in range(0, params["model"]["num_classes"]):
                    segmentation_mask[0, _class, ...] = global_postprocessing_dict[
                        postprocessor
                    ](segmentation_mask[0, _class, ...], params)

            return segmentation_mask

        def _swap_mask_axes_for_sikt_save_format_compatibility(
            segmentation_mask: np.ndarray,
        ):
            return np.swapaxes(segmentation_mask, 0, 2)

        def _postprocess_one_hot_reversed_segmentation_mask(
            segmentation_mask: np.ndarray, params: dict
        ):
            for postprocessor in params[
                "data_postprocessing_after_reverse_one_hot_encoding"
            ]:
                segmentation_mask = global_postprocessing_dict[postprocessor](
                    segmentation_mask, params
                )

            return segmentation_mask

        def _determine_final_prediction_mask_shape(segmentation_mask: np.ndarray):
            if segmentation_mask.shape[0] == 1:
                return segmentation_mask.squeeze(0)
            elif segmentation_mask.shape[-1] == 1:
                return segmentation_mask.squeeze(-1)
            else:
                return segmentation_mask

        predicted_segmentation_mask_numpy = predicted_segmentation_mask.numpy()
        predicted_segmentation_mask_numpy = _postprocess_raw_segmentation_mask(
            predicted_segmentation_mask_numpy, self.params
        )
        decoded_segmentation_mask = reverse_one_hot(
            predicted_segmentation_mask_numpy, self.params["model"]["class_list"]
        )
        decoded_segmentation_mask = _swap_mask_axes_for_sikt_save_format_compatibility(
            decoded_segmentation_mask
        )
        decoded_segmentation_mask = _postprocess_one_hot_reversed_segmentation_mask(
            decoded_segmentation_mask, self.params
        )
        decoded_segmentation_mask = _determine_final_prediction_mask_shape(
            decoded_segmentation_mask
        )

        image_save_format = get_filename_extension_sanitized(subject["1"]["path"][0])
        if image_save_format in [".jpg", ".jpeg", ".png"]:
            decoded_segmentation_mask = decoded_segmentation_mask.astype(np.uint8)

        subject_converted_to_sikt_format = _convert_subject_to_sikt_format(subject)
        result_sikt_image = sitk.GetImageFromArray(decoded_segmentation_mask)
        result_sikt_image.CopyInformation(subject_converted_to_sikt_format)

        if "resample" in self.params["data_preprocessing"]:
            result_sikt_image = resample_image(
                result_sikt_image,
                subject_converted_to_sikt_format.GetSpacing(),
                interpolator=sitk.sitkNearestNeighbor,
            )
        segmentation_mask_save_path = os.path.join(
            self._current_validation_epoch_save_dir,
            "testing",
            subject["subject_id"][0],
            f"{subject['subject_id'][0]}_seg_process_rank_{self.global_rank}{image_save_format}",
        )
        self._ensure_path_exists(os.path.dirname(segmentation_mask_save_path))
        sitk.WriteImage(result_sikt_image, segmentation_mask_save_path)

    def _save_predictions_for_regression_or_classification(
        self, rows_to_write: List[List[str]]
    ):
        csv_save_path = os.path.join(
            self._current_validation_epoch_save_dir, "output_predictions.csv"
        )
        file_contents_merged = self.CLASSIFICATION_REGRESSION_RESULTS_HEADER.join(
            [",".join(row) for row in rows_to_write]
        )
        with open(csv_save_path, "w") as file:
            file.write(file_contents_merged)

    def _check_if_early_stopping(self, val_loss):
        previous_best_loss = deepcopy(self.current_best_loss)
        if val_loss < self.current_best_loss:
            self.current_best_loss = val_loss
            self._save_model(self.current_epoch, self.model_paths["best"], False)
            print(
                f"Loss value improved. Previous best loss :{previous_best_loss}, new best loss: {val_loss} Saving best model from epoch {self.current_epoch}",
                flush=True,
            )
            self.wait_count_before_early_stopping = 0
        else:
            self.wait_count_before_early_stopping += 1
            print(
                f"Validation loss did not improve. Waiting count before early stopping: {self.wait_count_before_early_stopping} / {self.params['patience']}",
                flush=True,
            )
            if self.wait_count_before_early_stopping >= self.params["patience"]:
                self.trainer.should_stop = True
                print(
                    f"Early stopping triggered at epoch {self.current_epoch}, validation loss did not improve for {self.params['patience']} epochs, with the best loss value being {self.current_best_loss}. Stopping training.",
                    flush=True,
                )
        del previous_best_loss

    def on_test_start(self):
        self.test_metric_values: List[Dict[str, float]] = []
        self.test_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_test.csv"),
            metrics=list(self.params["metrics"]),
            mode="test",
        )

    def configure_optimizers(self):
        params = deepcopy(self.params)
        params["model_parameters"] = self.model.parameters()
        optimizer = get_optimizer(params)
        if "scheduler" in self.params:
            params["optimizer_object"] = optimizer
            scheduler = get_scheduler(params)
            return [optimizer], [scheduler]

        return optimizer

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = self._move_image_data_to_device(batch, device)
        batch = self._move_labels_or_values_to_device(batch, device)

        return batch

    def _move_image_data_to_device(self, subject, device):
        for channel_key in self.params["channel_keys"]:
            subject[channel_key][torchio.DATA] = subject[channel_key][torchio.DATA].to(
                device
            )

        return subject

    def _move_labels_or_values_to_device(self, subject, device):
        if "value_keys" in self.params:
            for value_key in self.params["value_keys"]:
                subject[value_key] = subject[value_key].to(device)
        else:
            subject["label"][torchio.DATA] = subject["label"][torchio.DATA].to(device)

        return subject
