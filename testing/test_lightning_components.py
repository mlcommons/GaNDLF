import os
import yaml
import torch
import math
import pytest
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from GANDLF.models.lightning_module import GandlfLightningModule
from GANDLF.losses.loss_calculators import (
    LossCalculatorFactory,
    LossCalculatorSimple,
    LossCalculatorSDNet,
    AbstractLossCalculator,
    LossCalculatorDeepSupervision,
)
from GANDLF.metrics.metric_calculators import (
    MetricCalculatorFactory,
    MetricCalculatorSimple,
    MetricCalculatorSDNet,
    MetricCalculatorDeepSupervision,
    AbstractMetricCalculator,
)
from GANDLF.utils.pred_target_processors import (
    PredictionTargetProcessorFactory,
    AbstractPredictionTargetProcessor,
    IdentityPredictionTargetProcessor,
    DeepSupervisionPredictionTargetProcessor,
)
from GANDLF.config_manager import ConfigManager
from GANDLF.parseConfig import parseConfig
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils.write_parse import parseTrainingCSV
from GANDLF.utils import populate_header_in_parameters, populate_channel_keys_in_params

from GANDLF.cli import patch_extraction

TESTS_DIRPATH = Path(__file__).parent.absolute().__str__()
TEST_DATA_DIRPATH = os.path.join(TESTS_DIRPATH, "data")
TEST_DATA_OUTPUT_DIRPATH = os.path.join(TESTS_DIRPATH, "data_output")
PATCH_SIZE = {"2D": [128, 128, 1], "3D": [32, 32, 32]}


def write_temp_config_path(parameters_to_write):
    print("02_2: Creating path for temporary config file")
    temp_config_path = os.path.join(TESTS_DIRPATH, "config_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    if parameters_to_write is not None:
        with open(temp_config_path, "w") as file:
            yaml.dump(parameters_to_write, file)
    return temp_config_path


class TrainerTestsContextManager:
    @staticmethod
    def _clear_output_dir(output_dir_path):
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
            os.makedirs(output_dir_path)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._clear_output_dir(TEST_DATA_OUTPUT_DIRPATH)


def add_mock_config_params(config):
    config["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
    config["model"]["class_list"] = [0, 1, 2, 3]


def read_config():
    config_path = Path(os.path.join(TESTS_DIRPATH, "config_segmentation.yaml"))

    csv_path = os.path.join(TEST_DATA_DIRPATH, "train_2d_rad_segmentation.csv")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    parsed_config = parseConfig(config)

    training_data, parsed_config["headers"] = parseTrainingCSV(csv_path)
    parsed_config = populate_header_in_parameters(
        parsed_config, parsed_config["headers"]
    )
    add_mock_config_params(parsed_config)
    return parsed_config


#### METRIC CALCULATORS ####


def test_port_pred_target_processor_identity():
    config = read_config()
    processor = PredictionTargetProcessorFactory(
        config
    ).get_prediction_target_processor()
    assert isinstance(
        processor, IdentityPredictionTargetProcessor
    ), f"Expected instance of {IdentityPredictionTargetProcessor}, got {type(processor)}"
    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    processed_preds, processed_target = processor(dummy_preds, dummy_target)
    assert torch.equal(dummy_preds, processed_preds)
    assert torch.equal(dummy_target, processed_target)


@pytest.mark.skip(
    reason="This is failing due to interpolation size mismatch - check it out"
)
def test_port_pred_target_processor_deep_supervision():
    config = read_config()
    config["model"]["architecture"] = "deep_supervision"
    processor = PredictionTargetProcessorFactory(
        config
    ).get_prediction_target_processor()
    assert isinstance(
        processor, DeepSupervisionPredictionTargetProcessor
    ), f"Expected instance of {DeepSupervisionPredictionTargetProcessor}, got {type(processor)}"
    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    processor(dummy_preds, dummy_target)


#### LOSS CALCULATORS ####


def test_port_loss_calculator_simple():
    config = read_config()
    processor = PredictionTargetProcessorFactory(
        config
    ).get_prediction_target_processor()
    loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
    assert isinstance(
        loss_calculator, LossCalculatorSimple
    ), f"Expected instance of {LossCalculatorSimple}, got {type(loss_calculator)}"

    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    processed_preds, processed_target = processor(dummy_preds, dummy_target)
    loss = loss_calculator(processed_preds, processed_target)
    assert not torch.isnan(loss).any()


def test_port_loss_calculator_sdnet():
    config = read_config()
    config["model"]["architecture"] = "sdnet"
    processor = PredictionTargetProcessorFactory(
        config
    ).get_prediction_target_processor()
    loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
    assert isinstance(
        loss_calculator, LossCalculatorSDNet
    ), f"Expected instance of {LossCalculatorSDNet}, got {type(loss_calculator)}"
    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    processed_preds, processed_target = processor(dummy_preds, dummy_target)
    loss = loss_calculator(processed_preds, processed_target)

    assert not torch.isnan(loss).any()


@pytest.mark.skip(
    reason="This is failing due to interpolation size mismatch - check it out"
)
def test_port_loss_calculator_deep_supervision():
    config = read_config()
    config["model"]["architecture"] = "deep_supervision"
    processor = PredictionTargetProcessorFactory(
        config
    ).get_prediction_target_processor()
    assert isinstance(
        loss_calculator, LossCalculatorDeepSupervision
    ), f"Expected instance of {LossCalculatorDeepSupervision}, got {type(loss_calculator)}"

    loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    processed_preds, processed_target = processor(dummy_preds, dummy_target)
    loss = loss_calculator(processed_preds, processed_target)
    assert not torch.isnan(loss).any()


#### METRIC CALCULATORS ####


def test_port_metric_calculator_simple():
    config = read_config()
    metric_calculator = MetricCalculatorFactory(config).get_metric_calculator()
    assert isinstance(
        metric_calculator, MetricCalculatorSimple
    ), f"Expected instance subclassing {MetricCalculatorSimple}, got {type(metric_calculator)}"
    dummy_preds = torch.randint(0, 4, (4, 4, 4, 4))
    dummy_target = torch.randint(0, 4, (4, 4, 4, 4))
    metric = metric_calculator(dummy_preds, dummy_target)
    for metric, value in metric.items():
        assert not math.isnan(value), f"Metric {metric} has NaN values"


def test_port_metric_calculator_sdnet():
    config = read_config()
    config["model"]["architecture"] = "sdnet"
    metric_calculator = MetricCalculatorFactory(config).get_metric_calculator()
    assert isinstance(
        metric_calculator, MetricCalculatorSDNet
    ), f"Expected instance of {MetricCalculatorSDNet}, got {type(metric_calculator)}"

    dummy_preds = torch.randint(0, 4, (1, 4, 4, 4, 4))
    dummy_target = torch.randint(0, 4, (4, 4, 4, 4))
    metric = metric_calculator(dummy_preds, dummy_target)
    for metric, value in metric.items():
        assert not math.isnan(value), f"Metric {metric} has NaN values"


@pytest.mark.skip(
    reason="This is failing due to interpolation size mismatch - check it out"
)
def test_port_metric_calculator_deep_supervision():
    config = read_config()
    config["model"]["architecture"] = "deep_supervision"
    metric_calculator = MetricCalculatorFactory(config).get_metric_calculator()
    assert isinstance(
        metric_calculator, MetricCalculatorDeepSupervision
    ), f"Expected instance of {MetricCalculatorDeepSupervision}, got {type(metric_calculator)}"

    dummy_preds = torch.randint(0, 4, (4, 4, 4, 4))
    dummy_target = torch.randint(0, 4, (4, 4, 4, 4))
    metric = metric_calculator(dummy_preds, dummy_target)
    for metric, value in metric.items():
        assert not math.isnan(value), f"Metric {metric} has NaN values"


#### LIGHTNING MODULE ####


def test_port_model_initialization():
    config = read_config()
    module = GandlfLightningModule(config, output_dir=TEST_DATA_OUTPUT_DIRPATH)
    assert module is not None, "Lightning module is None"
    assert module.model is not None, "Model architecture not initialized in the module"
    assert isinstance(
        module.loss, AbstractLossCalculator
    ), f"Expected instance subclassing  {AbstractLossCalculator}, got {type(module.loss)}"
    assert isinstance(
        module.metric_calculators, AbstractMetricCalculator
    ), f"Expected instance subclassing {AbstractMetricCalculator}, got {type(module.metric_calculators)}"
    assert isinstance(
        module.pred_target_processor, AbstractPredictionTargetProcessor
    ), f"Expected instance subclassing {AbstractPredictionTargetProcessor}, got {type(module.pred_target_processor)}"
    configured_optimizer, configured_scheduler = module.configure_optimizers()
    # In case of both optimizer and scheduler configured, lightning returns tuple of lists (optimizers, schedulers)
    # This is why I am checking for the first element of the iterable here
    configured_optimizer = configured_optimizer[0]
    configured_scheduler = configured_scheduler[0]
    assert isinstance(
        configured_optimizer, torch.optim.Optimizer
    ), f"Expected instance subclassing  {torch.optim.Optimizer}, got {type(configured_optimizer)}"
    assert isinstance(
        configured_scheduler, torch.optim.lr_scheduler.LRScheduler
    ), f"Expected instance subclassing  {torch.optim.lr_scheduler.LRScheduler}, got {type(configured_scheduler)}"


def test_port_model_2d_rad_segmentation_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_segmentation.yaml", version_check_flag=False
        )

        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_rad_segmentation.csv"
        )
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["2D"]
        parameters["metrics"].pop("iou")
        parameters["patience"] = 3
        parameters["model"]["dimension"] = 2
        parameters["model"]["class_list"] = [0, 255]
        parameters["model"]["amp"] = True
        parameters["model"]["num_channels"] = 3
        parameters["model"]["onnx_export"] = False
        parameters["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
        parameters["class_weights"] = [1.0, 1.0]
        parameters["sampling_weights"] = [1.0, 1.0]
        parameters["model"]["print_summary"] = True
        parameters["track_memory_usage"] = True
        parameters["verbose"] = True
        parameters["model"]["save_at_every_epoch"] = True
        parameters["save_output"] = True
        parameters = populate_header_in_parameters(parameters, parameters["headers"])

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        trainer.predict(module, test_dataloader)


def test_port_model_3d_rad_segmentation_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_segmentation.yaml", version_check_flag=False
        )

        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_3d_rad_segmentation.csv"
        )
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["3D"]
        parameters["metrics"].pop("iou")
        parameters["model"]["dimension"] = 3
        parameters["model"]["class_list"] = [0, 1]
        parameters["model"]["final_layer"] = "softmax"
        parameters["model"]["num_channels"] = len(
            parameters["headers"]["channelHeaders"]
        )
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["penalty_weights"] = [0.5, 0.25]
        parameters["class_weights"] = [1.0, 1.0]
        parameters["sampling_weights"] = [1.0, 1.0]
        parameters["track_memory_usage"] = True
        parameters["verbose"] = True
        parameters["model"]["save_at_every_epoch"] = True
        parameters["save_output"] = True
        parameters = populate_header_in_parameters(parameters, parameters["headers"])

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        trainer.predict(module, test_dataloader)


def test_port_model_2d_rad_regression_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_regression.yaml", version_check_flag=False
        )

        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_rad_regression.csv"
        )
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["2D"]
        parameters["model"]["dimension"] = 2
        parameters["model"]["amp"] = False
        parameters["model"]["num_channels"] = 3
        parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
        parameters["scaling_factor"] = 1
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["save_output"] = True
        parameters = populate_header_in_parameters(parameters, parameters["headers"])

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        trainer.predict(module, test_dataloader)


def test_port_model_3d_rad_regression_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_regression.yaml", version_check_flag=False
        )

        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_3d_rad_regression.csv"
        )
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["3D"]
        parameters["model"]["dimension"] = 3
        parameters["model"]["num_channels"] = len(
            parameters["headers"]["channelHeaders"]
        )
        parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
        parameters["scaling_factor"] = 1
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["save_output"] = True
        parameters = populate_header_in_parameters(parameters, parameters["headers"])

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        trainer.predict(module, test_dataloader)


def test_port_model_2d_rad_classification_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_classification.yaml", version_check_flag=False
        )
        parameters["modality"] = "rad"
        parameters["track_memory_usage"] = True
        parameters["patch_size"] = PATCH_SIZE["2D"]
        parameters["model"]["dimension"] = 2
        parameters["model"]["final_layer"] = "logits"
        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_rad_classification.csv"
        )
        parameters["model"]["num_channels"] = 3
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["save_output"] = True
        parameters["model"]["architecture"] = "densenet121"
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        inference_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_rad_classification.csv"
        )
        inference_data.drop("ValueToPredict", axis=1, inplace=True)
        inference_data.drop("Label", axis=1, inplace=True)
        temp_infer_csv = os.path.join(TEST_DATA_OUTPUT_DIRPATH, "temp_infer_csv.csv")
        inference_data.to_csv(temp_infer_csv, index=False)
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_classification.yaml", version_check_flag=False
        )
        inference_data, parameters["headers"] = parseTrainingCSV(temp_infer_csv)
        parameters["output_dir"] = TEST_DATA_OUTPUT_DIRPATH  # this is in inference mode
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["2D"]
        parameters["model"]["dimension"] = 2
        parameters["model"]["final_layer"] = "logits"
        parameters["model"]["num_channels"] = 3
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        parameters["model"]["architecture"] = "densenet121"
        parameters["model"]["onnx_export"] = False
        parameters["differential_privacy"] = False
        parameters["save_output"] = True

        dataset = ImagesFromDataFrame(
            inference_data, parameters, train=False, loader_type="testing"
        )

        inference_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        parameters = populate_channel_keys_in_params(inference_dataloader, parameters)

        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.predict(module, inference_dataloader)


# TODO Refactor this and other tests
def test_port_model_3d_rad_classification_single_device_single_node(device):
    with TrainerTestsContextManager():
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_classification.yaml", version_check_flag=False
        )
        parameters["modality"] = "rad"
        parameters["track_memory_usage"] = True
        parameters["patch_size"] = PATCH_SIZE["3D"]
        parameters["model"]["dimension"] = 3
        parameters["model"]["final_layer"] = "logits"
        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_3d_rad_classification.csv"
        )
        parameters["model"]["num_channels"] = len(
            parameters["headers"]["channelHeaders"]
        )
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["model"]["architecture"] = "densenet121"
        parameters["save_output"] = True
        parameters = populate_header_in_parameters(parameters, parameters["headers"])

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )
        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        training_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_3d_rad_classification.csv"
        )
        training_data.drop("ValueToPredict", axis=1, inplace=True)
        training_data.drop("Label", axis=1, inplace=True)
        temp_infer_csv = os.path.join(TEST_DATA_OUTPUT_DIRPATH, "temp_infer_csv.csv")
        training_data.to_csv(temp_infer_csv, index=False)
        parameters = parseConfig(
            TESTS_DIRPATH + "/config_classification.yaml", version_check_flag=False
        )
        training_data, parameters["headers"] = parseTrainingCSV(temp_infer_csv)
        parameters["output_dir"] = TEST_DATA_OUTPUT_DIRPATH  # this is in inference mode
        parameters["modality"] = "rad"
        parameters["patch_size"] = PATCH_SIZE["3D"]
        parameters["model"]["dimension"] = 3
        parameters["model"]["final_layer"] = "logits"
        parameters["model"]["num_channels"] = len(
            parameters["headers"]["channelHeaders"]
        )
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        parameters["model"]["architecture"] = "densenet121"
        parameters["model"]["onnx_export"] = False
        parameters["differential_privacy"] = False
        parameters["save_output"] = True

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="testing"
        )

        inference_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        parameters = populate_channel_keys_in_params(inference_dataloader, parameters)

        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.predict(module, inference_dataloader)


def test_port_model_classification_histology_2d_single_device_single_node(device):
    with TrainerTestsContextManager():
        output_dir_patches = os.path.join(TEST_DATA_OUTPUT_DIRPATH, "histo_patches")
        if os.path.isdir(output_dir_patches):
            shutil.rmtree(output_dir_patches)
        Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
        output_dir_patches_output = os.path.join(
            output_dir_patches, "histo_patches_output"
        )

        parameters_patch = {}
        # extracting minimal number of patches to ensure that the test does not take too long
        parameters_patch["patch_size"] = [128, 128]

        for num_patches in [-1, 3]:
            parameters_patch["num_patches"] = num_patches
            file_config_temp = write_temp_config_path(parameters_patch)

            if os.path.exists(output_dir_patches_output):
                shutil.rmtree(output_dir_patches_output)
            # this ensures that the output directory for num_patches=3 is preserved
            Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)
            patch_extraction(
                TEST_DATA_DIRPATH + "/train_2d_histo_classification.csv",
                output_dir_patches_output,
                file_config_temp,
            )

        file_for_Training = os.path.join(output_dir_patches_output, "opm_train.csv")
        temp_df = pd.read_csv(file_for_Training)
        temp_df.drop("Label", axis=1, inplace=True)
        temp_df["valuetopredict"] = np.random.randint(2, size=6)
        temp_df.to_csv(file_for_Training, index=False)
        # read and parse csv
        parameters = ConfigManager(
            TESTS_DIRPATH + "/config_classification.yaml", version_check_flag=False
        )
        parameters["modality"] = "histo"
        parameters["patch_size"] = 128
        file_config_temp = write_temp_config_path(parameters)
        parameters = ConfigManager(file_config_temp, version_check_flag=False)
        os.remove(file_config_temp)
        parameters["model"]["dimension"] = 2
        # read and parse csv
        training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
        parameters["model"]["num_channels"] = 3
        parameters["model"]["architecture"] = "densenet121"
        parameters["model"]["norm_type"] = "none"
        parameters["data_preprocessing"]["rgba2rgb"] = ""
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        parameters["nested_training"]["testing"] = 1
        parameters["nested_training"]["validation"] = -2
        parameters["model"]["print_summary"] = False
        parameters["model"]["onnx_export"] = False
        parameters["differential_privacy"] = False

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )

        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        inference_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_histo_classification.csv", train=False
        )
        trainer.predict(module, inference_data.iterrows())


def test_port_model_segmentation_histology_2d_single_device_single_node(device):
    with TrainerTestsContextManager():
        output_dir_patches = os.path.join(TEST_DATA_OUTPUT_DIRPATH, "histo_patches")
        if os.path.isdir(output_dir_patches):
            shutil.rmtree(output_dir_patches)
        Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
        output_dir_patches_output = os.path.join(
            output_dir_patches, "histo_patches_output"
        )
        Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)

        parameters_patch = {}
        # extracting minimal number of patches to ensure that the test does not take too long
        parameters_patch["num_patches"] = 10
        parameters_patch["read_type"] = "sequential"
        # define patches to be extracted in terms of microns
        parameters_patch["patch_size"] = ["1000m", "1000m"]

        file_config_temp = write_temp_config_path(parameters_patch)

        patch_extraction(
            TEST_DATA_DIRPATH + "/train_2d_histo_segmentation.csv",
            output_dir_patches_output,
            file_config_temp,
        )
        os.remove(file_config_temp)
        file_for_Training = os.path.join(output_dir_patches_output, "opm_train.csv")
        parameters = ConfigManager(
            TESTS_DIRPATH + "/config_segmentation.yaml", version_check_flag=False
        )
        training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
        parameters["patch_size"] = PATCH_SIZE["2D"]
        parameters["modality"] = "histo"
        parameters["model"]["dimension"] = 2
        parameters["model"]["class_list"] = [0, 255]
        parameters["penalty_weights"] = [1, 1]
        parameters["model"]["amp"] = True
        parameters["model"]["num_channels"] = 3
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        parameters["model"]["architecture"] = "resunet"
        parameters["nested_training"]["testing"] = 1
        parameters["nested_training"]["validation"] = -2
        parameters["metrics"] = ["dice"]
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = True
        parameters["data_preprocessing"]["resize_image"] = [128, 128]

        dataset = ImagesFromDataFrame(
            training_data, parameters, train=True, loader_type="train"
        )
        dataset_val = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="validation"
        )
        dataset_test = ImagesFromDataFrame(
            training_data, parameters, train=False, loader_type="test"
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters["batch_size"], shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, batch_size=parameters["batch_size"], shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=parameters["batch_size"], shuffle=False
        )

        parameters = populate_channel_keys_in_params(train_dataloader, parameters)
        module = GandlfLightningModule(parameters, output_dir=TEST_DATA_OUTPUT_DIRPATH)
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,
            num_nodes=1,
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloader, val_dataloader)
        trainer.test(module, test_dataloader)

        inference_data, parameters["headers"] = parseTrainingCSV(
            TEST_DATA_DIRPATH + "/train_2d_histo_segmentation.csv"
        )
        inference_data.drop(index=inference_data.index[-1], axis=0, inplace=True)

        trainer.predict(module, inference_data.iterrows())
