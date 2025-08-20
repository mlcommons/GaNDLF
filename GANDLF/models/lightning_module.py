import os
import sys
import time
import psutil
import torch
import torchio
import warnings
import openslide
import numpy as np
import SimpleITK as sitk
import lightning.pytorch as pl
import torch.nn.functional as F


from medcam import medcam
from copy import deepcopy
from statistics import mean
from multiprocessing import Lock
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from GANDLF.logger import Logger
from GANDLF.models import get_model
from GANDLF.metrics import overall_stats
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.data.post_process import global_postprocessing_dict
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.metrics.metric_calculators import MetricCalculatorFactory
from GANDLF.data.preprocessing import get_transforms_for_preprocessing
from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset
from GANDLF.utils.pred_target_processors import PredictionTargetProcessorFactory
from GANDLF.privacy.opacus.opacus_anonymization_manager import (
    OpacusAnonymizationManager,
)
from GANDLF.privacy.opacus import handle_dynamic_batch_size


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
    MapSaver,
)

from typing import Tuple, Union, Dict, List, Any


class GandlfLightningModule(pl.LightningModule):
    CLASSIFICATION_REGRESSION_RESULTS_HEADER = "Epoch,SubjectID,PredictedValue\n"
    CLASSIFICATION_REGRESSION_RESULTS_HEADER_HISTOPATH = "SubjectID,x_coords,y_coords"
    FLOAT_FORMATTING_PRECISION = 4
    MULTIPROCESSING_LOCK = Lock()

    def __init__(self, params: dict, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.params = deepcopy(params)
        self.learning_rate = self.params["learning_rate"]
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
        """
        Creates the BaseModel instance based on the parameters.
        """

        self.model = get_model(self.params)

    def _initialize_loss(self):
        """
        Initializes the loss calculator based on the parameters. Loss calculator
        logic differs for some specific model architectures, see the LossCalculatorFactory
        for more details.
        """

        self.loss = LossCalculatorFactory(self.params).get_loss_calculator()

    def _initialize_metric_calculators(self):
        """
        Initializes the metric calculators based on the parameters. Metric calculators
        logic differs for some specific model architectures, see the MetricCalculatorFactory
        for more details.
        """

        self.metric_calculators = MetricCalculatorFactory(
            self.params
        ).get_metric_calculator()

    def _initialize_preds_target_processor(self):
        """Initializes the prediction target processor based on the parameters.
        This processor ensures that the prediction and target tensors are in the correct format,
        as some architectures may require different formats for the predictions and targets.
        """

        self.pred_target_processor = PredictionTargetProcessorFactory(
            self.params
        ).get_prediction_target_processor()

    def _initialize_model_save_paths(self):
        """
        Initializes the paths used for saving checkpoints of the model.
        """

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
    def _save_model(self, epoch: int, save_path: str, onnx_export: bool):
        """
        Saves the model to the specified path, adhering to GANDLF save format.

        Args:
            epoch (int): The epoch number.
            save_path (str): The path to save the model to.
            onnx_export (bool): Whether to export the model to ONNX format
        """
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

    def _prepare_metrics_dict_for_progbar_logging(
        self, metric_results_dict: Dict[str, float]
    ):
        """
        Formats the metric results dictionary into format suitable for
        logging with Lightning's progress bar.

        Args:
            metric_results_dict (Dict[str, float]): The dictionary containing the metric results.
        """
        metric_results_dict_with_updated_suffix = (
            self._add_stage_prefix_to_metric_results_dict(
                metric_results_dict, self._determine_trainer_stage_string()
            )
        )
        metric_results_dict_with_values_formatted = (
            self._convert_per_class_metric_results_to_separate_key_value_pairs(
                metric_results_dict_with_updated_suffix
            )
        )
        return self._round_metric_values_in_dict(
            metric_results_dict_with_values_formatted
        )

    @staticmethod
    def _convert_per_class_metric_results_to_separate_key_value_pairs(
        metric_results_dict: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        In case the metric results dictionary contains per-class values, this function
        takes the values and creates separate key-value pairs for each class in the
        results dictionary.

        Args:
            metric_results_dict (Dict[str, Any]): The dictionary containing the metric results.

        Returns:
            parsed_results_dict (Dict[str, float]): The dictionary containing the parsed results.
        """
        parsed_results_dict = deepcopy(metric_results_dict)
        for metric_name, metric_value in metric_results_dict.items():
            if isinstance(metric_value, list):
                for n, metric_value_for_given_class in enumerate(metric_value):
                    parsed_results_dict[
                        metric_name + f"_class_{n}"
                    ] = metric_value_for_given_class
                del parsed_results_dict[metric_name]
        return parsed_results_dict

    @staticmethod
    def _add_stage_prefix_to_metric_results_dict(
        metric_results_dict: Dict[str, float], stage: str
    ):
        """
        Ensures that metric names in the results dictionary are prefixed with the stage
        """
        metric_results_dict_with_updated_suffix = {
            f"{stage}_{metric_name}": metric_value
            for metric_name, metric_value in metric_results_dict.items()
        }
        return metric_results_dict_with_updated_suffix

    def _round_metric_values_in_dict(self, metric_results_dict: Dict[str, float]):
        """
        Performs rounding of the metric values in the results dictionary.

        Args:
            metric_results_dict (Dict[str, float]): The dictionary containing the metric results.

        Returns:
            rounded_metric_results_dict (Dict[str, float]): The dictionary containing the rounded metric results.
        """

        return {
            k: self._round_value_to_precision(v) for k, v in metric_results_dict.items()
        }

    def _round_value_to_precision(self, value: float):
        """
        Rounds the value to the specified precision, defined as module constant.

        Args:
            value (float): The value to round.

        Returns:
            rounded_value (float): The rounded value.
        """

        return round(value, self.FLOAT_FORMATTING_PRECISION)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Forward pass of the model.
        """
        attention_map = None
        is_medcam_enabled = self.params.get("medcam_enabled", False)
        if is_medcam_enabled:
            output, attention_map = self.model(images)
            if self.params["model"]["dimension"] == 2:
                attention_map = torch.unsqueeze(attention_map, -1)
        else:
            output = self.model(images)
        return output, attention_map

    def on_train_start(self):
        self._print_training_initialization_info()
        self._set_training_start_time()
        self._print_channels_info()
        self._initialize_train_logger()
        self._initialize_training_epoch_containers()
        self.wait_count_before_early_stopping = 0
        self.current_best_loss = sys.float_info.max

        self.params["current_epoch"] = self.current_epoch
        # TODO check out if the disabled by default medcam is indeed what we
        # meant - it was taken from original code
        if self.params.get("medcam"):
            self._inject_medcam_module()
            self.params["medcam_enabled"] = False
        if self.params.get("differential_privacy"):
            self._initialize_training_differential_privacy()

    def _try_to_load_model_training_start(self):
        """
        Attempts to load the model at the start of the training.
        """
        if self._try_to_load_model(self.model_paths["best"]):
            print(f"Previous best model loaded from {self.model_paths['best']}.")
        elif self._try_to_load_model(self.model_paths["latest"]):
            print(f"Previous latest model loaded from {self.model_paths['latest']}.")
        else:
            print(
                "Could not load any previous model, training from scratch.", flush=True
            )

    def _try_to_load_model(self, load_path: str):
        """
        Attempts to load the model from the specified path.

        Args:
            load_path (str): The path to the model to load.

        Returns:
            bool: Whether the model was successfully loaded.
        """
        if os.path.exists(load_path):
            try:
                checkpoint_dict = load_model(load_path, self.device)
                version_check(
                    self.params["version"], version_to_check=checkpoint_dict["version"]
                )
                # I am purposefully omitting the line below, as "previous_parameters" are not used anywhere
                # params["previous_parameters"] = main_dict.get("parameters", None)
                state_dict = checkpoint_dict["model_state_dict"]
                if self.params.get("differential_privacy"):
                    # this is required for torch==1.11 and for DP inference
                    new_state_dict = {}
                    for key, val in state_dict.items():
                        new_key = key.replace("_module.", "")
                        new_state_dict[new_key] = val  # remove `module.`
                    state_dict = new_state_dict

                self.model.load_state_dict(state_dict)
                if self.trainer.training:
                    self.optimizers(False).load_state_dict(
                        checkpoint_dict["optimizer_state_dict"]
                    )
                self.trainer.fit_loop.epoch_progress.current.completed = (
                    checkpoint_dict["epoch"]
                )
                self.trainer.callback_metrics["val_loss"] = checkpoint_dict["loss"]
                return True
            except Exception as e:
                warnings.warn(
                    f"Model found under path {load_path}, but error occurred during loading: {e}"
                )
        return False

    @rank_zero_only
    def _try_to_save_initial_model(self):
        """
        Saves the initial model at the specified path if it does not already exist.
        """
        if not os.path.exists(self.model_paths["initial"]):
            self._save_model(self.current_epoch, self.model_paths["initial"], False)
            print(f"Initial model saved at {self.model_paths['initial']}")
        else:
            print(
                f"Initial model already exists at {self.model_paths['initial']}; Skipping saving"
            )

    def _inject_medcam_module(self):
        """
        Extends the model with the medcam module, used for generating attention maps.
        """
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

    def _get_metrics_names_for_loggers(self):
        """
        Returns the names of the overall metrics to be logged if the problem type is classification or regression.
        """
        metric_names = list(self.params["metrics"])
        overall_metrics = {}
        if self._problem_type_is_regression:
            overall_metrics = overall_stats(
                torch.tensor([1]), torch.tensor([1]), self.params
            )
        elif self._problem_type_is_classification:
            temp_tensor = torch.randint(
                0, self.params["model"]["num_classes"], (5,), dtype=torch.int32
            )
            overall_metrics = overall_stats(temp_tensor, temp_tensor, self.params)
        for overall_metric_key in overall_metrics.keys():
            if overall_metric_key not in metric_names:
                metric_names.append(overall_metric_key)

        return metric_names

    @rank_zero_only
    def _initialize_train_logger(self):
        self.train_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_training.csv"),
            metrics=self._get_metrics_names_for_loggers(),
            mode="train",
        )

    @rank_zero_only
    def _set_training_start_time(self):
        self.training_start_time = time.time()

    @rank_zero_only
    def _print_training_initialization_info(self):
        """
        Basic info printed at the start of the training.
        """
        self._print_host_info()
        if self.params["verbose"]:
            print("Initializing training at :", get_date_time(), flush=True)
        if self.params["model"]["print_summary"]:
            self._print_model_summary()

    def _print_host_info(self):
        if os.environ.get("HOSTNAME"):
            print("Hostname :", os.environ.get("HOSTNAME"), flush=True)

    def _print_model_summary(self):
        print_model_summary(
            self.model,
            self.params["batch_size"],
            self.params["model"]["num_channels"],
            self.params["patch_size"],
        )

    @rank_zero_only
    def _initialize_training_epoch_containers(self):
        """
        Initializes the containers for storing the training epoch data.
        They are used for accumulating the losses, metrics, predictions and labels
        for each epoch, so final calculations can be made at the end of the epoch.
        """

        self.train_losses: List[torch.Tensor] = []
        self.training_metric_values: List[Dict[str, float]] = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_predictions: List[torch.Tensor] = []
            self.train_labels: List[torch.Tensor] = []

    @rank_zero_only
    def _print_channels_info(self):
        print("Number of channels : ", self.params["model"]["num_channels"])

    def training_step(self, subject, batch_idx):
        """
        Single training optimization step.
        """
        if self.params.get("save_training"):
            write_training_patches(subject, self.params)

        if self.params.get("differential_privacy"):
            self._handle_dynamic_batch_size_in_differential_privacy_mode(subject)

        images = self._prepare_images_batch_from_subject_data(subject)
        labels = self._prepare_labels_batch_from_subject_data(subject)

        images = self._ensure_proper_images_tensor_dimensions(images)
        labels = self._process_labels(labels)
        model_output, _ = self.forward(images)
        model_output, labels = self.pred_target_processor(model_output, labels)

        loss = self.loss(model_output, labels, images)
        metric_results = self.metric_calculators(
            model_output, labels, subject_spacing=subject.get("spacing", None)
        )

        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_labels.append(labels.detach().cpu())
            self.train_predictions.append(
                torch.argmax(model_output, dim=1).detach().cpu()
            )

        self.train_losses.append(loss.detach().cpu())
        self.training_metric_values.append(metric_results)

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
        """
        Creates the label tensor from the subject data.

        Args:
            subject (torchio.Subject): The torchio.Subject object containing the label.

        Returns:
            label (torch.Tensor): The label tensor of shape (B, C, H, W, D) for segmentation,
            or a tensor of shape (B, ) for classification/regression.
        """

        if self._problem_type_is_regression or self._problem_type_is_classification:
            label = torch.cat(
                [subject[key] for key in self.params["value_keys"]], dim=0
            )
            # TODO this for sure needs some further investigation
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(self.params["batch_size"], len(label)),
                len(self.params["value_keys"]),
            )
        else:
            label = subject["label"][torchio.DATA]

        return label

    def _ensure_proper_images_tensor_dimensions(self, images: torch.Tensor):
        """
        Modify the input images by removing the singular depth dimension added
        by torchio for 2D images.

        Args:
            images (torch.Tensor): The input images tensor.

        Returns:
            images (torch.Tensor): The modified images tensor.
        """

        if self.params["model"]["dimension"] == 2:
            images = images.squeeze(-1)

        return images

    def _process_labels(self, labels: torch.Tensor):
        """
        Modifies the labels tensor based on the problem type.
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
        subject, _ = handle_dynamic_batch_size(subject, self.params)
        return subject

    def _initialize_training_differential_privacy(self):
        self._check_if_opacus_is_applicable()
        opacus_manager = OpacusAnonymizationManager(self.params)

        (
            model,
            dp_optimizer,
            train_dataloader,
            privacy_engine,
        ) = opacus_manager.apply_privacy(
            self.model, self.optimizers().optimizer, self.trainer.train_dataloader
        )
        self.model = model
        self.trainer.fit_loop._data_source.instance = train_dataloader
        self.trainer.optimizers = [dp_optimizer]
        # TODO should we reinit the scheduler too?
        self._dp_engine = privacy_engine

    def _check_if_opacus_is_applicable(self):
        if isinstance(self.trainer.strategy, DDPStrategy):
            raise NotImplementedError(
                "Differential privacy is not supported with DDP strategy. Please use single GPU."
            )

    def on_train_epoch_start(self):
        self._set_epoch_start_time()
        if self.params["track_memory_usage"]:
            self._write_epoch_start_process_resource_usage(self.current_epoch)
        if self.params["verbose"]:
            self._print_epoch_start_time()

    def _write_epoch_start_process_resource_usage(self, epoch):
        """
        Writes the memory usage to a file at the start of the epoch.
        Ran separately on each process in case of distributed training.

        Args:
            epoch (int): The current epoch number.
        """
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

        # TODO evaluate if this indeed works properly in distributed setting
        self.MULTIPROCESSING_LOCK.acquire()
        with open(full_filepath, file_write_mode) as file_mem:
            file_mem.write(memory_info_string)
        self.MULTIPROCESSING_LOCK.release()

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

    # TODO check if it indeed work properly and run only on rank 0
    @rank_zero_only
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
        mean_loss = self._round_value_to_precision(
            torch.mean(torch.stack(self.train_losses)).item()
        )

        self._clear_training_epoch_containers()

        self.train_logger.write(
            self.current_epoch,
            mean_loss,
            self._ensure_proper_metric_formatting_for_logging(epoch_metrics),
        )
        self.log("train_loss", mean_loss, on_epoch=True, prog_bar=True)
        self.log_dict(
            self._prepare_metrics_dict_for_progbar_logging(epoch_metrics),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.params["verbose"]:
            self._print_epoch_end_time()
        if self.params["model"]["save_at_every_epoch"]:
            self._save_epoch_end_checkpoint()
        if os.path.exists(self.model_paths["latest"]):
            os.remove(self.model_paths["latest"])
        self._save_model(self.current_epoch, self.model_paths["latest"], False)

        print("Latest model saved")

    def _compute_metric_mean_across_values_from_batches(
        self, metric_values: List[Union[float, List[float]]]
    ) -> Union[float, List[float]]:
        """
        Given a list of metrics calculated for each batch, computes the mean across all batches.
        Takes into account case where metric is a list of values (e.g. for each class).

        Args:
            metric_values (List[Union[float, List[float]]]): The list of metric values for each batch.

        Returns:
            Union[float, List[float]]: The mean value of the metric across all batches.
        """
        if isinstance(metric_values[0], list):
            return [
                mean([batch_metrics[i] for batch_metrics in metric_values])
                for i in range(len(metric_values[0]))
            ]
        return self._round_value_to_precision(mean(metric_values))

    @staticmethod
    def _ensure_proper_metric_formatting_for_logging(metrics_dict: dict) -> dict:
        """
        Helper function to ensure that all metric values are in the correct format for
        GANDLF's logging system.

        Args:
            metrics_dict (dict): The dictionary containing the metric values.

        Returns:
            output_metrics_dict (dict): The dictionary containing the formatted metric values.
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

    @rank_zero_only
    def _save_epoch_end_checkpoint(self):
        """
        Saves the model at the end of the epoch.
        """
        epoch_save_path = os.path.join(
            self.output_dir,
            self.params["model"]["architecture"]
            + "_epoch_"
            + str(self.current_epoch)
            + ".pth.tar",
        )
        self._save_model(self.current_epoch, epoch_save_path, False)
        print("Epoch model saved.")

    @rank_zero_only
    def _print_epoch_end_time(self):
        print(
            "Time taken for epoch : ",
            (time.time() - self.epoch_start_time) / 60,
            " mins",
            flush=True,
        )

    @rank_zero_only
    def _clear_training_epoch_containers(self):
        self.train_losses = []
        self.training_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.train_predictions = []
            self.train_labels = []

    @rank_zero_only
    def on_train_end(self):
        if os.path.exists(self.model_paths["best"]):
            # Why don't we handle it here with the full save_model function?
            # TODO Onnx export seems to modify model INPLACE, so when doing cuda
            optimize_and_save_model(
                self.model, self.params, self.model_paths["best"], onnx_export=False
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

    @rank_zero_only
    def on_validation_start(self):
        self._initialize_validation_epoch_containers()
        self._initialize_validation_logger()

    @rank_zero_only
    def _initialize_validation_epoch_containers(self):
        self.val_losses: List[torch.Tensor] = []
        self.validation_metric_values: List[Dict[str, float]] = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.val_predictions: List[float] = []
            self.val_labels: List[float] = []
            if self.params["save_output"]:
                self.rows_to_write: List[str] = []

    @rank_zero_only
    def _initialize_validation_logger(self):
        self.val_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_validation.csv"),
            metrics=self._get_metrics_names_for_loggers(),
            mode="val",
            add_epsilon=bool(self.params.get("differential_privacy")),
        )

    @rank_zero_only
    def on_validation_epoch_start(self):
        # TODO this is dead code both here and in original loops
        # by default medcam is injected at the training and ["medcam_enabled"] is set to False
        # so this block is never executed
        if self.params["medcam_enabled"]:
            self.model.enable_medcam()
            self.params["medcam_enabled"] = True
        self._current_validation_epoch_save_dir = os.path.join(
            self.output_dir, "output_validation", f"epoch_{self.current_epoch}"
        )
        self._ensure_path_exists(self._current_validation_epoch_save_dir)

    def validation_step(self, subject, batch_idx):
        if self.params["verbose"]:
            self._print_currently_processed_subject(subject)

        subject_dict = self._initialize_subject_dict_nontraining_mode(subject)
        label_present = subject["label"] != ["NA"]
        value_keys_present = "value_keys" in self.params
        label = None
        if label_present:
            subject_dict = self._extend_nontraining_subject_dict_with_label(
                subject, subject_dict
            )

        if (
            self._problem_type_is_regression
            or self._problem_type_is_classification
            and label_present
        ):
            (
                model_output,
                last_input_batch,
            ) = self._get_predictions_on_subject_using_label_sampler(subject_dict)

            if self.params["save_output"]:
                processed_logit = self._process_prediction_logit_for_row_writing(
                    model_output, self.params["scaling_factor"]
                )
                self.rows_to_write.append(
                    self._prepare_row_for_output_csv(
                        subject["subject_id"][0], processed_logit, self.current_epoch
                    )
                )

            label = self._initialize_nontraining_label_ground_truth_classification_or_regression(
                subject
            )
        else:
            (
                model_output,
                last_input_batch,
            ) = self._get_predictions_on_subject_using_grid_sampler(subject_dict)

            if self.params["save_output"]:
                self._save_predictions_for_segmentation_subject(model_output, subject)

            if self._problem_type_is_segmentation and label_present:
                label = self._initialize_nontraining_label_ground_truth_segmentation(
                    subject
                )
            elif (
                self._problem_type_is_classification
                or self._problem_type_is_regression
                and value_keys_present
            ):
                label = self._initialize_nontraining_label_ground_truth_classification_or_regression(
                    subject
                )

        if label is not None:
            label = self._process_labels(label)
            model_output, label = self.pred_target_processor(model_output, label)
            loss = self.loss(model_output, label, last_input_batch)
            metric_results = self.metric_calculators(
                model_output, label, subject_spacing=subject.get("spacing", None)
            )

            self.val_losses.append(loss)
            self.validation_metric_values.append(metric_results)

        if (
            self._problem_type_is_regression
            or self._problem_type_is_classification
            and label
        ):
            model_prediction = (
                torch.argmax(model_output[0], 0)
                if self._problem_type_is_classification
                else model_output[0]
            )  # TODO am I right here? For regression, we should not take argmax
            self.val_predictions.append(model_prediction.item())
            self.val_labels.append(label.item())

    @staticmethod
    def _prepare_row_for_output_csv(
        subject_id: str, prediction_logit: float, epoch: int
    ):
        """
        Helper function to prepare the row for the output CSV file.

        Args:
            subject_id (str): The subject ID.
            prediction_logit (float): The prediction logit.
            epoch (int): The epoch number.

        Returns:
            row (str): The row to write to the output CSV file.
        """

        return f"{epoch},{subject_id},{prediction_logit}\n"

    @staticmethod
    def _prepare_row_for_output_csv_histopathology_inference(
        subject_name, x_coord, y_coord, output_matrix
    ):
        """
        Helper function to prepare the row for the output CSV file in histopathology inference.

        Args:
            subject_name (str): The subject name.
            x_coord (int): The x coordinate.
            y_coord (int): The y coordinate.
            output_matrix (np.array) : output matrix of the model, a set of
        predicted 2D matrices for each class

        Returns:
            row (str): The row to write to the output CSV file.
        """
        base_string = f"{subject_name},{x_coord},{y_coord}"
        for output_for_class in output_matrix:
            base_string += f",{output_for_class}"
        return base_string + "\n"

    @staticmethod
    def _process_prediction_logit_for_row_writing(
        prediction_logit: torch.Tensor, scaling_factor: float = 1.0
    ):
        """
        Processes the prediction logits for writing to the output CSV file.

        Args:
            prediction_logit (torch.Tensor): The prediction logits.
            scaling_factor (float): The scaling factor modifying the prediction logit.
            Default is 1 (no scaling).

        Returns:
            prediction_logit (float): The processed prediction logit.
        """
        return prediction_logit.cpu().max().item() / scaling_factor

    def _print_currently_processed_subject(self, subject):
        if isinstance(subject, torchio.Subject):
            subject_id = subject["subject_id"]
        elif isinstance(subject, tuple):
            # ugly corner histology inference handling, when incoming batch is
            # a row from dataframe, not a torchio.Subject. This should be solved
            # via some kind of polymorphism in the future
            subject_data = subject[1]
            subject_id = subject_data[self.params["headers"]["subjectIDHeader"]]
        print("== Current subject:", subject_id, flush=True)

    def _initialize_subject_dict_nontraining_mode(self, subject: torchio.Subject):
        """
        Create a dictionary containing the subject data for the non-training mode
        (validation, testing, inference).

        Args:
            subject (torchio.Subject): The subject data.

        Returns:
            subject_dict (Dict[str, torchio.Image]): The dictionary containing the subject data.
        """
        subject_dict = {}

        for channel_key in self.params["channel_keys"]:
            subject_dict[channel_key] = torchio.ScalarImage(
                path=subject[channel_key]["path"],
                tensor=subject[channel_key]["data"].squeeze(0),
                affine=subject[channel_key]["affine"].squeeze(0),
            )
        value_keys_present = "value_keys" in self.params
        if (
            self._problem_type_is_regression
            or self._problem_type_is_classification
            and value_keys_present
        ):
            for key in self.params["value_keys"]:
                subject_dict["value_" + key] = subject[key]

        return subject_dict

    def _extend_nontraining_subject_dict_with_label(
        self, subject: torchio.Subject, subject_dict: dict
    ) -> dict:
        """
        Extends the subject dictionary with the label data for the non-training mode.

        Args:
            subject (torchio.Subject): The subject data.
            subject_dict (dict): The dictionary containing the subject data.

        Returns:
            subject_dict (dict): The dictionary containing the subject data with the label data.
        """
        subject_dict["label"] = torchio.LabelMap(
            path=subject["label"]["path"],
            tensor=subject["label"]["data"].squeeze(0),
            affine=subject["label"]["affine"].squeeze(0),
        )

        return subject_dict

    def _initialize_nontraining_label_ground_truth_classification_or_regression(
        self, subject: torchio.Subject
    ):
        """
        Initializes the ground truth label for classification or regression problems
        in the non-training mode (validation, testing, inference).

        Args:
            subject_dict (torchio.Subject): The dictionary containing the subject data.

        Returns:
            label (torch.Tensor): The ground truth label tensor.
        """
        return torch.cat([subject[key] for key in self.params["value_keys"]], dim=0)

    def _initialize_nontraining_label_ground_truth_segmentation(
        self, subject: torchio.Subject
    ):
        """
        Initializes the ground truth label for segmentation problems in the non-training mode
        (validation, testing, inference).

        Args:
            subject_dict (torchio.Subject): The dictionary containing the subject data.

        Returns:
            label (torch.Tensor): The ground truth label tensor
        """

        return subject["label"]["data"]

    # TODO this whole logic can be packed into something separate, as it is only used
    # in validation of regression and classification problems
    def _get_predictions_on_subject_using_label_sampler(
        self, subject_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on the subject using the label sampler. Used for regression and classification problems.

        Args:
            subject_dict (dict): The dictionary containing the subject data.

        Returns:
            total_logits_for_all_patches (torch.Tensor): The total logits for all patches
        extracted from a subject, normalized by the number of samples per volume.
            last_batch_of_input_images (torch.Tensor): The last batch of input images. Used
        mostly for special cases like deep_resunet, deep_unet, etc. when it is needed for
        loss calculation.
        """

        def _prepare_images_batch_from_patch_regression_or_classification_with_label_sampler(
            patches_batch: torchio.Subject,
        ):
            """
            Sampling the patches using the label sampler requires a different approach
            to preparing the images batch (concatenation dimension changes compared to logic
            in other steps).

            Args:
                patches_batch (torchio.Subject): The batch of patches for the subject.

            Returns:
                images_batch_from_patches (torch.Tensor): The images batch from the patches.
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

        model_outputs_list: List[torch.Tensor] = []
        for patches_batch in patch_loader:
            images_from_patches = _prepare_images_batch_from_patch_regression_or_classification_with_label_sampler(
                patches_batch
            )
            images_from_patches = self._ensure_proper_images_tensor_dimensions(
                images_from_patches
            )
            model_output, _ = self.forward(images_from_patches)
            model_outputs_list.append(model_output)

        total_logits_for_all_patches = torch.cat(model_outputs_list).sum(
            dim=0, keepdim=True
        )
        return (
            total_logits_for_all_patches / self.params["q_samples_per_volume"],
            images_from_patches,
        )

    @rank_zero_only
    def _determine_trainer_stage_string(self):
        """
        Helper function to determine the trainer stage and store it as a module attribute.
        """
        if self.trainer.validating:
            return "val"
        elif self.trainer.testing:
            return "test"
        elif self.trainer.predicting:
            return "inference"

        return "train"

    def _determine_save_path_to_use(self):
        """
        Helper function to determine the output save path based on the trainer stage.
        """
        if self.trainer.validating:
            return self._current_validation_epoch_save_dir
        elif self.trainer.testing:
            return self._current_test_epoch_save_dir
        elif self.trainer.predicting:
            return self._current_inference_save_dir
        else:
            raise RuntimeError("Output save path cannot be determined for training")

    def _get_predictions_on_subject_using_grid_sampler(
        self, subject_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on the subject using the grid sampler. This is used in segmentation
        problems in validation and testing and for all problems in inference
        (as no ground truth is available in inference).

        Args:
            subject_dict (dict): The dictionary containing the subject data.

        Returns:
            aggregated_predictions (torch.Tensor): The predicted segmentation mask.
            last_batch_of_input_images (torch.Tensor): The last batch of input images. Used
        mostly for special cases like deep_resunet, deep_unet, etc. when it is needed for
        loss calculation.
        """

        def _ensure_output_is_tensor_for_special_architectures(
            model_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ):
            """
            Helper function to ensure that the output is a tensor for special architectures
            that return a tuple of tensors (SDnet, DeepResunet etc)

            Args:
                model_output (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The model output.
            """

            if not isinstance(model_output, torch.Tensor):
                warnings.warn(
                    f"Model output is not a Tensor: {type(model_output)}. Say, `deep_resunet` and `deep_unet` may return "
                    f"list of tensors on different scales instead of just one prediction Tensor. However due to "
                    f"GaNDLF architecture it is expected that models return only one tensor. For deep_* models "
                    f"only the biggest scale is processed. Use these models with caution till fix is implemented."
                )
                model_output = model_output[0]

            return model_output

        def _ensure_output_shape_compatibility_with_torchio(model_output: torch.Tensor):
            """
            Helper function to ensure that the output shape is compatible with torchio (4D for 2D segmentation).

            Args:
                model_output (torch.Tensor): The model output tensor.

            Returns:
                model_output (torch.Tensor): The model output tensor with the correct shape.
            """
            if (
                self.params["model"]["dimension"] == 2
                and self._problem_type_is_segmentation
            ):
                model_output = model_output.unsqueeze(-1)
            return model_output

        grid_sampler = self._prepare_grid_sampler(subject_dict)
        patch_loader = self._prepare_dataloader_from_grid_sampler(grid_sampler)

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
        if self._problem_type_is_regression or self._problem_type_is_classification:
            model_outputs_list: List[torch.Tensor] = []

        for patches_batch in patch_loader:
            images_from_patches = self._prepare_images_batch_from_subject_data(
                patches_batch
            )
            images_from_patches = self._ensure_proper_images_tensor_dimensions(
                images_from_patches
            )
            model_output, attention_map = self.forward(images_from_patches)

            model_output = _ensure_output_is_tensor_for_special_architectures(
                model_output
            )
            model_output = _ensure_output_shape_compatibility_with_torchio(model_output)
            if self.params["medcam_enabled"]:
                medcam_attention_map_aggregator.add_batch(
                    attention_map, patches_batch[torchio.LOCATION]  # type: ignore
                )
            if self._problem_type_is_segmentation:
                prediction_aggregator.add_batch(
                    model_output, patches_batch[torchio.LOCATION]
                )
            else:
                model_outputs_list.append(model_output)

        if self.params["medcam_enabled"]:
            attention_map = medcam_attention_map_aggregator.get_output_tensor()
            for i, n in enumerate(attention_map):
                self.model.save_attention_map(
                    n.squeeze(), raw_input=images_from_patches[i].squeeze(-1)
                )

        if self._problem_type_is_regression or self._problem_type_is_classification:
            return (
                torch.cat(model_outputs_list).sum(dim=0, keepdim=True)
                / len(patch_loader),
                images_from_patches,
            )

        return (
            prediction_aggregator.get_output_tensor().unsqueeze(0).to(self.device),
            images_from_patches,
        )

    def _prepare_grid_sampler(self, subject_dict: dict):
        """
        Creates the grid sampler for the grid aggregator.

        Args:
            subject_dict (dict): The dictionary containing the subject data.

        Returns:
            grid_sampler (torchio.inference.GridSampler): The grid sampler.
        """
        grid_sampler = torchio.inference.GridSampler(
            torchio.Subject(subject_dict),
            self.params["patch_size"],
            patch_overlap=self.params["inference_mechanism"]["patch_overlap"],
        )
        return grid_sampler

    def _prepare_dataloader_from_grid_sampler(
        self, grid_sampler: torchio.inference.GridSampler
    ):
        """
        Creates the dataloader from the grid sampler.

        Args:
            grid_sampler (torchio.inference.GridSampler): The grid sampler.

        Returns:
            patch_loader (torch.utils.data.DataLoader): The patch loader.
        """

        return torch.utils.data.DataLoader(grid_sampler, batch_size=1)  # type: ignore

    # TODO check if it indeed work properly and run only on rank 0
    @rank_zero_only
    def on_validation_epoch_end(self):
        validation_epoch_average_metrics = {}
        metric_names = self.validation_metric_values[0].keys()
        for metric_name in metric_names:
            metric_values = [x[metric_name] for x in self.validation_metric_values]
            validation_epoch_average_metrics[
                metric_name
            ] = self._compute_metric_mean_across_values_from_batches(metric_values)

        if self._problem_type_is_regression or self._problem_type_is_classification:
            # This is a workaround - sometimes the lists are empty
            preds_or_labels_not_empty = not (
                len(self.val_predictions) == 0 or len(self.val_labels) == 0
            )
            if preds_or_labels_not_empty:
                validation_epoch_average_metrics_overall = overall_stats(
                    torch.tensor(self.val_predictions),
                    torch.tensor(self.val_labels),
                    self.params,
                )
                validation_epoch_average_metrics.update(
                    validation_epoch_average_metrics_overall
                )
        mean_loss = self._round_value_to_precision(
            torch.mean(torch.stack(self.val_losses)).item()
        )

        self.val_logger.write(
            self.current_epoch,
            mean_loss,
            self._ensure_proper_metric_formatting_for_logging(
                validation_epoch_average_metrics
            ),
        )

        self.log("val_loss", mean_loss, on_epoch=True, prog_bar=True)
        self.log_dict(
            self._prepare_metrics_dict_for_progbar_logging(
                validation_epoch_average_metrics
            ),
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        self._check_if_early_stopping(mean_loss)

        if self.params["save_output"] and (
            self._problem_type_is_regression or self._problem_type_is_classification
        ):
            self._save_predictions_csv_for_regression_or_classification(
                self.rows_to_write, self._determine_save_path_to_use()
            )
        if self.params.get("differential_privacy"):
            self._print_differential_privacy_info()
        self._clear_validation_epoch_containers()

    @rank_zero_only
    def _clear_validation_epoch_containers(self):
        self.val_losses = []
        self.validation_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.val_predictions = []
            self.val_labels = []
            if self.params["save_output"]:
                self.rows_to_write = []

    @rank_zero_only
    def _ensure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @rank_zero_only
    def _print_differential_privacy_info(self):
        delta = self.params["differential_privacy"]["delta"]
        epsilon = self._dp_engine.get_epsilon(delta)
        print(f"Epoch {self.current_epoch} Privacy: ε = {epsilon:.2f}, δ = {delta}")
        self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        self.log("delta", delta, on_epoch=True, prog_bar=True)

    # TODO called at the validation step, NOT at the end of the epoch - we want to avoid
    # saving all predictions for all subjects for the end of the epoch
    def _save_predictions_for_segmentation_subject(
        self, predicted_segmentation_mask: torch.Tensor, subject: torchio.Subject
    ):
        """
        Saves the predicted segmentation mask for a given subject, performing the necessary postprocessing
        steps.

        Args:
            predicted_segmentation_mask (torch.Tensor): The predicted segmentation mask, extracted
        from the grid aggregator when all validation patches for this subject have
        been processed.
            subject (torchio.Subject): The subject for which the segmentation mask was predicted, used
        to extract the metadata.
        """

        def _convert_subject_to_sitk_format(subject: torchio.Subject):
            return torchio.ScalarImage(
                tensor=subject["1"]["data"].squeeze(0).cpu(),
                affine=subject["1"]["affine"].squeeze(0).cpu(),
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

        def _swap_mask_axes_for_sitk_save_format_compatibility(
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

        predicted_segmentation_mask_numpy = predicted_segmentation_mask.cpu().numpy()
        predicted_segmentation_mask_numpy = _postprocess_raw_segmentation_mask(
            predicted_segmentation_mask_numpy, self.params
        )
        # taking 0-th element as the batch size is 1, and this is required by reverse_one_hot function
        decoded_segmentation_mask = reverse_one_hot(
            predicted_segmentation_mask_numpy[0], self.params["model"]["class_list"]
        )
        decoded_segmentation_mask = _swap_mask_axes_for_sitk_save_format_compatibility(
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

        subject_converted_to_sitk_format = _convert_subject_to_sitk_format(subject)
        result_sitk_image = sitk.GetImageFromArray(decoded_segmentation_mask)
        result_sitk_image.CopyInformation(subject_converted_to_sitk_format)

        if "resample" in self.params["data_preprocessing"]:
            result_sitk_image = resample_image(
                result_sitk_image,
                subject_converted_to_sitk_format.GetSpacing(),
                interpolator=sitk.sitkNearestNeighbor,
            )
        segmentation_mask_save_path = os.path.join(
            self._determine_save_path_to_use(),
            subject["subject_id"][0],
            f"{subject['subject_id'][0]}_seg_process_rank_{self.global_rank}{image_save_format}",
        )
        self._ensure_path_exists(os.path.dirname(segmentation_mask_save_path))
        sitk.WriteImage(result_sitk_image, segmentation_mask_save_path)

    @rank_zero_only
    def _save_predictions_csv_for_regression_or_classification(
        self, rows_to_write: List[str], save_path: str
    ):
        """
        Saves the predictions for regression or classification problems to a CSV file.

        Args:
            rows_to_write (List[str]): The rows to write to the CSV file. Each element of
        the list is a row.
            save_path (str): The save path for the CSV file.
        """

        def _determine_header_to_use():
            if self.trainer.predicting:
                if self.params["modality"] in ["histo", "path"]:
                    header = self.CLASSIFICATION_REGRESSION_RESULTS_HEADER_HISTOPATH
                    if self._problem_type_is_regression:
                        return header + ",output\n"
                    elif self._problem_type_is_classification:
                        for class_num in range(self.params["model"]["num_classes"]):
                            header += f",probability_{class_num}"
                        return header + "\n"
            return self.CLASSIFICATION_REGRESSION_RESULTS_HEADER

        csv_save_path = os.path.join(save_path, "output_predictions.csv")
        merged_output = _determine_header_to_use()
        for row in rows_to_write:
            merged_output += row
        with open(csv_save_path, "w") as file:
            file.write(merged_output)

    # TODO separate it into checking and saving functions, perhaps even separate class
    @rank_zero_only
    def _check_if_early_stopping(self, val_loss: float):
        """
        Checks if early stopping should be triggered based on the validation loss.
        If the loss improves, the best model is saved.
        """
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
            if self.wait_count_before_early_stopping > self.params["patience"]:
                self.trainer.should_stop = True
                print(
                    f"Early stopping triggered at epoch {self.current_epoch}, validation loss did not improve for {self.params['patience']} epochs, with the best loss value being {self.current_best_loss}. Stopping training.",
                    flush=True,
                )
        del previous_best_loss

    def on_test_start(self):
        self._initialize_test_epoch_containers()
        self._initialize_test_logger()

    @rank_zero_only
    def _initialize_test_logger(self):
        self.test_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_test.csv"),
            metrics=self._get_metrics_names_for_loggers(),
            mode="test",
        )

    @rank_zero_only
    def _initialize_test_epoch_containers(self):
        self.test_losses: List[torch.Tensor] = []
        self.test_metric_values: List[Dict[str, float]] = []

    def on_test_epoch_start(self):
        if self.params["medcam_enabled"]:
            self.model.enable_medcam()
            self.params["medcam_enabled"] = True

        self._current_test_epoch_save_dir = os.path.join(
            self.output_dir, "output_test", f"epoch_{self.current_epoch}"
        )
        self._ensure_path_exists(self._current_test_epoch_save_dir)

    def test_step(self, subject, batch_idx):
        if self.params["verbose"]:
            self._print_currently_processed_subject(subject)

        subject_dict = self._initialize_subject_dict_nontraining_mode(subject)
        label_present = subject["label"] != ["NA"]
        value_keys_present = "value_keys" in self.params
        label = None
        if label_present:
            subject_dict = self._extend_nontraining_subject_dict_with_label(
                subject, subject_dict
            )
        if (
            self._problem_type_is_regression
            or self._problem_type_is_classification
            and label_present
        ):
            (
                model_output,
                last_input_batch,
            ) = self._get_predictions_on_subject_using_label_sampler(subject_dict)

            if self.params["save_output"]:
                processed_logit = self._process_prediction_logit_for_row_writing(
                    model_output, self.params["scaling_factor"]
                )
                self.rows_to_write.append(
                    self._prepare_row_for_output_csv(
                        subject["subject_id"][0], processed_logit, self.current_epoch
                    )
                )

            label = self._initialize_nontraining_label_ground_truth_classification_or_regression(
                subject
            )
        else:
            (
                model_output,
                last_input_batch,
            ) = self._get_predictions_on_subject_using_grid_sampler(subject_dict)
            if self.params["save_output"]:
                self._save_predictions_for_segmentation_subject(model_output, subject)
            if self._problem_type_is_segmentation and label_present:
                label = self._initialize_nontraining_label_ground_truth_segmentation(
                    subject
                )
            elif (
                self._problem_type_is_classification
                or self._problem_type_is_regression
                and value_keys_present
            ):
                label = self._initialize_nontraining_label_ground_truth_classification_or_regression(
                    subject
                )
        if label is not None:
            label = self._process_labels(label)
            model_output, label = self.pred_target_processor(model_output, label)

            loss = self.loss(model_output, label, last_input_batch)
            metric_results = self.metric_calculators(
                model_output, label, subject_spacing=subject.get("spacing", None)
            )

            self.test_losses.append(loss)
            self.test_metric_values.append(metric_results)

    @rank_zero_only
    def on_test_epoch_end(self):
        test_epoch_average_metrics = {}
        metric_names = self.test_metric_values[0].keys()
        for metric_name in metric_names:
            metric_values = [x[metric_name] for x in self.test_metric_values]
            test_epoch_average_metrics[
                metric_name
            ] = self._compute_metric_mean_across_values_from_batches(metric_values)

        mean_loss = self._round_value_to_precision(
            torch.mean(torch.stack(self.test_losses)).item()
        )

        self.test_logger.write(
            self.current_epoch,
            mean_loss,
            self._ensure_proper_metric_formatting_for_logging(
                test_epoch_average_metrics
            ),
        )

        self.log("test_loss", mean_loss, on_epoch=True, prog_bar=True)
        self.log_dict(
            self._prepare_metrics_dict_for_progbar_logging(test_epoch_average_metrics),
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        if self.params["save_output"] and (
            self._problem_type_is_regression or self._problem_type_is_classification
        ):
            self._save_predictions_csv_for_regression_or_classification(
                self.rows_to_write, self._determine_save_path_to_use()
            )
        self._clear_test_epoch_containers()

    @rank_zero_only
    def _clear_test_epoch_containers(self):
        self.test_losses = []
        self.test_metric_values = []

    def on_predict_start(self):
        self._initialize_inference_containers()
        self._try_to_load_model_inference_start()

        if self.params.get("differential_privacy"):
            self._initialize_inference_differential_privacy()

    def _try_to_load_model_inference_start(self):
        if self._try_to_load_model(self.model_paths["best"]):
            print(f"Previous best model loaded from {self.model_paths['best']}.")
        elif self._try_to_load_model(self.model_paths["latest"]):
            print(f"Previous latest model loaded from {self.model_paths['latest']}.")
        else:
            raise RuntimeError(
                f"Best/latest models not found to load: {self.model_paths}"
            )

    @rank_zero_only
    def _initialize_inference_containers(self):
        self._current_inference_save_dir = os.path.join(
            self.output_dir, "output_inference"
        )  # TODO here we need some mechanism for separate outputs for nested inference
        self._ensure_path_exists(self._current_inference_save_dir)
        self.inference_losses = []
        self.inference_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.rows_to_write = []
            self.subject_classification_class_probabilities: Dict[
                str, torch.Tensor
            ] = {}

    @rank_zero_only
    def _print_inference_initialization_info(self):
        print("Current model type : ", self.params["model"]["type"])
        print("Number of dims     : ", self.params["model"]["dimension"])
        if "num_channels" in self.params["model"]:
            print("Number of channels : ", self.params["model"]["num_channels"])
        print("Number of classes  : ", len(self.params["model"]["class_list"]))
        self._print_host_info()
        if self.params["model"]["print_summary"]:
            self._print_model_summary()

    def predict_step(self, batch, batch_idx):
        if self.params["verbose"]:
            self._print_currently_processed_subject(batch)
        # TODO both of those below should return values to complete the logic
        # of calculating metrics for classification case that is currently handled
        # by saving/reading logits.csv file
        if self.params["modality"] == "rad":
            return self._radiology_inference_step(batch)
        else:
            return self._histopathology_inference_step(batch)

    def _radiology_inference_step(self, subject: torchio.Subject):
        label_present = subject["label"] != ["NA"]
        subject_dict = self._initialize_subject_dict_nontraining_mode(subject)
        if label_present:
            subject_dict = self._extend_nontraining_subject_dict_with_label(
                subject, subject_dict
            )
            if (
                self._problem_type_is_regression
                or self._problem_type_is_classification
                and label_present
            ):
                (
                    model_output,
                    last_input_batch,
                ) = self._get_predictions_on_subject_using_label_sampler(subject_dict)

                processed_logit = self._process_prediction_logit_for_row_writing(
                    model_output, self.params["scaling_factor"]
                )
                self.rows_to_write.append(
                    self._prepare_row_for_output_csv(
                        subject["subject_id"][0], processed_logit, self.current_epoch
                    )
                )

                label = self._initialize_nontraining_label_ground_truth_classification_or_regression(
                    subject
                )
            else:
                (
                    model_output,
                    last_input_batch,
                ) = self._get_predictions_on_subject_using_grid_sampler(subject_dict)
                self._save_predictions_for_segmentation_subject(model_output, subject)
                label = self._initialize_nontraining_label_ground_truth_segmentation(
                    subject
                )
            label = self._process_labels(label)
            model_output, label = self.pred_target_processor(model_output, label)

            loss = self.loss(model_output, label, last_input_batch)
            metric_results = self.metric_calculators(
                model_output, label, subject_spacing=subject.get("spacing", None)
            )

            self.inference_losses.append(loss)
            self.inference_metric_values.append(metric_results)
        else:
            (
                model_output,
                last_input_batch,
            ) = self._get_predictions_on_subject_using_grid_sampler(subject_dict)
            if self._problem_type_is_classification or self._problem_type_is_regression:
                processed_logit = self._process_prediction_logit_for_row_writing(
                    model_output
                )
                self.rows_to_write.append(
                    self._prepare_row_for_output_csv(
                        subject["subject_id"][0], processed_logit, self.current_epoch
                    )
                )
            else:
                self._save_predictions_for_segmentation_subject(model_output, subject)

        if self._problem_type_is_classification:
            self.subject_classification_class_probabilities[
                subject["subject_id"][0]
            ] = F.softmax(model_output, dim=1)

    # TODO this has to be somehow handled in different way, we
    # are mixing too much logic in this single module
    def _histopathology_inference_step(self, row_index_tuple):
        """
        Inference step for the histopathology modality. This function is called with an assumption that the highest
        level dataloader is an iterator over the rows of the dataframe. The function is called for each row of the
        dataframe.

        Args:
            row (pd.Series): The row of the dataframe containing the information about the slide to be processed.

        """
        row = row_index_tuple[1]
        subject_name = row[self.params["headers"]["subjectIDHeader"]]
        inference_results_save_dir_for_subject = os.path.join(
            self._current_inference_save_dir, "histopathology", str(subject_name)
        )
        self._ensure_path_exists(inference_results_save_dir_for_subject)
        self._prepare_histopath_default_inference_params()
        openslide_image = openslide.open_slide(
            row[self.params["headers"]["channelHeaders"]].values[0]
        )
        max_defined_slide_level = openslide_image.level_count - 1
        row_slide_level = min(self.params["slide_level"], max_defined_slide_level)
        row_slide_level = min(row_slide_level, 0)
        level_width, level_height = openslide_image.level_dimensions[row_slide_level]
        patch_size = self._ensure_patch_size_is_2D(self.params["patch_size"])
        count_map = self._initialize_count_map(level_width, level_height)
        probabilities_map = self._initialize_probability_map(
            self.params["model"]["num_classes"], level_width, level_height
        )

        # TODO this should be done by other object or method
        transform_requested = get_transforms_for_preprocessing(
            self.params, [], False, False
        )
        patient_dataset = InferTumorSegDataset(
            row[self.params["headers"]["channelHeaders"]].values[0],
            patch_size=patch_size,
            stride_size=self.params["stride_size"],
            selected_level=row_slide_level,
            mask_level=self.params["mask_level"],
            transform=transform_requested,
        )
        histopathology_dataloader = torch.utils.data.DataLoader(
            patient_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.params["q_num_workers"],
        )
        patch_size_updated_after_transforms = patient_dataset.get_patch_size()
        if self.params["model"]["print_summary"]:
            print_model_summary(
                self.model,
                self.params["batch_size"],
                self.params["model"]["num_channels"],
                patch_size_updated_after_transforms,
            )
        count_map, probabilities_map = self._iterate_over_histopathology_loader(
            histopathology_dataloader,
            count_map,
            probabilities_map,
            patch_size_updated_after_transforms,
            self.params["model"]["num_classes"],
            subject_name,
        )

        map_saver = MapSaver(
            num_classes=self.params["model"]["num_classes"],
            slide_level=row_slide_level,
            blending_alpha=self.params["blending_alpha"],
            level_height=level_height,
            level_width=level_width,
        )
        map_saver.save_count_map(
            count_map, save_dir=inference_results_save_dir_for_subject
        )

        map_saver.save_probability_and_segmentation_maps(
            probabilities_map,
            openslide_image,
            save_dir=inference_results_save_dir_for_subject,
        )
        if self._problem_type_is_classification or self._problem_type_is_regression:
            self._save_predictions_csv_for_regression_or_classification(
                self.rows_to_write, inference_results_save_dir_for_subject
            )

    def _iterate_over_histopathology_loader(
        self,
        histopathology_dataloader,
        count_map,
        probability_map,
        patch_size,
        num_classes,
        subject_name,
    ):
        for image_patches, (x_coord, y_coord) in histopathology_dataloader:
            x_coord, y_coord = (
                x_coord.numpy(),
                y_coord.numpy(),
            )  # TODO the dataset should do that when fetching
            image_patches = image_patches.to(self.device)
            output, _ = self.forward(image_patches)
            output = output.cpu().detach().numpy()
            for i in range(output.shape[0]):
                self._increment_value_of_count_map_at_given_position(
                    count_map, x_coord[i], y_coord[i], patch_size
                )
                for class_index in range(num_classes):
                    self._add_value_to_probability_map_at_given_position(
                        probability_map,
                        x_coord[i],
                        y_coord[i],
                        patch_size,
                        output[i][class_index],
                        class_index,
                    )
                if (
                    self._problem_type_is_regression
                    or self._problem_type_is_classification
                ):
                    row_for_csv_saving = (
                        self._prepare_row_for_output_csv_histopathology_inference(
                            subject_name, x_coord[i], y_coord[i], output[i]
                        )
                    )
                    self.rows_to_write.append(row_for_csv_saving)
        probability_map = np.divide(probability_map, count_map)
        return count_map, probability_map

    @staticmethod
    def _increment_value_of_count_map_at_given_position(
        count_map, x_coord, y_coord, patch_size
    ):
        count_map[
            y_coord : y_coord + patch_size[1], x_coord : x_coord + patch_size[0]
        ] += 1

    @staticmethod
    def _add_value_to_probability_map_at_given_position(
        prob_map, x_coord, y_coord, patch_size, value, class_index
    ):
        prob_map[
            class_index,
            y_coord : y_coord + patch_size[1],
            x_coord : x_coord + patch_size[0],
        ] += value

    # TODO this should be handled by the config parser
    @rank_zero_only
    def _prepare_histopath_default_inference_params(self):
        """
        Sets the parameters necessary for histopath inference.
        """
        self.params["stride_size"] = self.params.get("stride_size", None)
        self.params["slide_level"] = self.params.get("slide_level", 0)
        self.params["mask_level"] = self.params.get(
            "mask_level", self.params["slide_level"]
        )
        self.params["blending_alpha"] = float(self.params.get("blending_alpha", 0.5))

    @staticmethod
    def _initialize_count_map(level_width: int, level_height: int):
        """
        Initializes the count maps for the histopathology inference.

        Args:
            level_width (int): The width of the level.
            level_height (int): The height of the level.

        Returns:
            count_map (np.ndarray): The count map.
        """
        return np.zeros((level_height, level_width), dtype=np.uint8)

    @staticmethod
    def _initialize_probability_map(
        num_classes: int, level_width: int, level_height: int
    ):
        """
        Initializes the probability maps for the histopathology inference.
        Called for classification and segmentation problems.

        Args:
            num_classes (int): The number of classes.
            level_width (int): The width of the level.
            level_height (int): The height of the level.

        Returns:
            probs_map (np.ndarray): The probability map.
        """
        return np.zeros((num_classes, level_height, level_width), dtype=np.float16)

    @staticmethod
    def _ensure_patch_size_is_2D(patch_size: List[int]):
        """
        Ensures that the patch size is 2D.

        Args:
            patch_size (List[int]): The patch size.

        Returns:
            patch_size (List[int]): The 2D patch size.
        """
        if len(patch_size) == 3:
            return patch_size[:-1]
        return patch_size

    @rank_zero_only
    def on_predict_end(self):
        if self.inference_metric_values:
            inference_epoch_average_metrics = {}
            metric_names = self.inference_metric_values[0].keys()
            for metric_name in metric_names:
                metric_values = [x[metric_name] for x in self.inference_metric_values]
                inference_epoch_average_metrics[
                    metric_name
                ] = self._compute_metric_mean_across_values_from_batches(metric_values)

            mean_loss = self._round_value_to_precision(
                torch.mean(torch.stack(self.inference_losses)).item()
            )

            print("Inference results:")
            print(f"Loss: {mean_loss}")
            print(f"Metrics: {inference_epoch_average_metrics}")

        self._clear_inference_containers()

    @rank_zero_only
    def _clear_inference_containers(self):
        self.inference_losses = []
        self.inference_metric_values = []
        if self._problem_type_is_regression or self._problem_type_is_classification:
            self.rows_to_write = []

    def configure_optimizers(self):
        params = deepcopy(self.params)
        params["model_parameters"] = self.model.parameters()
        params["learning_rate"] = self.learning_rate
        optimizer = get_optimizer(params)
        if "scheduler" in self.params:
            params["optimizer_object"] = optimizer
            scheduler = get_scheduler(params)
            optimizer_dict = {"optimizer": optimizer, "scheduler": scheduler}
            if isinstance(scheduler, ReduceLROnPlateau):
                optimizer_dict["monitor"] = "val_loss"
            return optimizer_dict
        return {"optimizer": optimizer}

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        A method called by Lightning to transfer the batch to the device.
        In case of GANDLF, we need custom logic to transfer the data to the device.
        """
        if not (
            self.trainer.predicting and self.params["modality"] in ["path", "histo"]
        ):
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
        elif subject["label"] != ["NA"]:
            subject["label"][torchio.DATA] = subject["label"][torchio.DATA].to(device)

        return subject
