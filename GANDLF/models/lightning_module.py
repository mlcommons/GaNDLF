import os
import time
import psutil
import torch
import torchio
import warnings
from medcam import medcam
from copy import deepcopy
from statistics import mean
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import EarlyStopping
from GANDLF.logger import Logger
from GANDLF.models import get_model
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.metrics.metric_calculators import MetricCalculatorFactory
from GANDLF.utils.pred_target_processors import PredictionTargetProcessorFactory

from GANDLF.utils import (
    optimize_and_save_model,
    write_training_patches,
    one_hot,
    print_model_summary,
    get_date_time,
    save_model,
    load_model,
    version_check,
    BEST_MODEL_PATH_END,
    INITIAL_MODEL_PATH_END,
    LATEST_MODEL_PATH_END,
)

from overrides import override
from typing import Tuple, Union, Dict, List


class GandlfLightningModule(pl.LightningModule):
    def __init__(self, params: dict, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.params = deepcopy(params)
        self.current_best_loss = sys.float_info.max
        self.wait_count_before_early_stopping = 0
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
                    f"Previous best model found under path {self.model_paths['best']}, but error occured during loading: {e}; Continuing training with new model"
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
        self.train_losses: List[torch.Tensor] = []
        self.training_metric_values: List[Dict[str, float]] = []
        self.train_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_training.csv"),
            metrics=list(self.params["metrics"]),
            mode="train",
        )
        self.val_logger = Logger(
            logger_csv_filename=os.path.join(self.output_dir, "logs_validation.csv"),
            metrics=list(self.params["metrics"]),
            mode="val",
            add_epsilon=bool(self.params.get("differential_privacy")),
        )
        if "medcam" in self.params:
            self._enable_medcam()

        if "differential_privacy" in self.params:
            self._initialize_differential_privacy()

    def _enable_medcam(self):
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
        # Should it really be set to false here? Seems like we are forcing it to be disabled
        # as in later stages we are checking if it is true or not
        self.params["medcam_enabled"] = False

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
        images, labels = self._process_inputs(images, labels)

        model_output, _ = self.forward(images)
        model_output, labels = self.pred_target_processor(model_output, labels)
        loss = self.loss(model_output, labels)
        metric_results = self.metric_calculators(model_output, labels)

        self.train_losses.append(loss.detach().cpu())
        self.training_metric_values.append(metric_results)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metric_results, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def _prepare_images_batch_from_subject_data(self, subject):
        images_batch = torch.cat(  # 5D tensor: (B,C, H, W, D)
            [subject[key][torchio.DATA] for key in self.params["channel_keys"]], dim=1
        )
        return images_batch

    def _prepare_labels_batch_from_subject_data(self, subject):
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

        return label

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

    # TODO when used with multple GPUs, thil will produce multiple logs
    # for each GPU. We should think on doing allgather here in a function
    # that is called on the main process (rank 0)

    def on_train_epoch_end(self):
        training_epoch_average_metrics = {}
        metric_names = self.training_metric_values[0].keys()
        for metric_name in metric_names:
            metric_values = [x[metric_name] for x in self.training_metric_values]
            training_epoch_average_metrics[metric_name] = mean(metric_values)

        train_losses_gathered = self.all_gather(self.train_losses)
        mean_loss = torch.mean(torch.stack(train_losses_gathered)).item()
        # Note that we DO NOT have overall stats calculators here as in the original code
        # Is it valid? Should we add it? This woudl require accumulation of all the predictions and labels
        # thus greatly increasing memory usage, especially in segmentation tasks
        self.train_logger.write(
            self.current_epoch, mean_loss, training_epoch_average_metrics
        )
        self.log_dict(training_epoch_average_metrics, on_epoch=True, prog_bar=True)
        self.training_metric_values = []
        self.train_losses = []
        if self.params["verbose"]:
            self._print_epoch_end_time()
        if self.params["model"]["save_at_every_epoch"]:
            epoch_save_path = os.path.join(
                self.output_dir,
                self.params["model"]["architecture"]
                + "_epoch_"
                + str(self.current_epoch)
                + ".pth.tar",
            )
            self._save_model(self.current_epoch, epoch_save_path, False)
            print(f"Epoch model saved.")
        if os.path.exists(self.model_paths["latest"]):
            os.remove(self.model_paths["latest"])
        self._save_model(self.current_epoch, self.model_paths["latest"], False)
        print(f"Latest model saved")

    @rank_zero_only
    def _print_epoch_end_time(self):
        print(
            "Time taken for epoch : ",
            (time.time() - self.epoch_start_time) / 60,
            " mins",
            flush=True,
        )

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
        self.val_losses: List[torch.Tensor] = []
        self.validation_metric_values: List[Dict[str, float]] = []

    # TODO placeholder for now
    def validation_step(self, subject, batch_idx):
        loss = torch.randn([1])
        self.val_losses.append(loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # TODO to be extended
    def on_validation_epoch_end(self):
        val_losses_gathered = self.all_gather(self.val_losses)
        mean_loss = torch.mean(torch.stack(val_losses_gathered)).item()
        self._check_if_early_stopping(mean_loss)

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
