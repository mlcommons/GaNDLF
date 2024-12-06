import os
import yaml
import torch
import math
import pytest
from pathlib import Path
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
from GANDLF.parseConfig import parseConfig
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils.write_parse import parseTrainingCSV
from GANDLF.utils import populate_header_in_parameters, populate_channel_keys_in_params

TESTS_DIRPATH = Path(__file__).parent.absolute().__str__()
TEST_DATA_DIRPATH = os.path.join(TESTS_DIRPATH, "data")
PATCH_SIZE = {"2D": [128, 128, 1], "3D": [32, 32, 32]}


def add_mock_config_params(config):
    config["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
    config["model"]["class_list"] = [0, 1, 2, 3]


def read_config():
    config_path = Path(os.path.join(TESTS_DIRPATH, "config_segmentation.yaml"))

    csv_path = os.path.join(TESTS_DIRPATH, "data/train_2d_rad_segmentation.csv")
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

    dummy_preds = torch.randint(0, 4, (4, 4, 4, 4))
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
    module = GandlfLightningModule(config, output_dir=TESTS_DIRPATH)
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


def test_port_model_forward_2d_rad_segmentation_single_gpu_single_node(device):
    parameters = parseConfig(
        TESTS_DIRPATH + "/config_segmentation.yaml", version_check_flag=False
    )

    training_data, parameters["headers"] = parseTrainingCSV(
        TEST_DATA_DIRPATH + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = PATCH_SIZE["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
    parameters["class_weights"] = [1.0, 1.0]
    parameters["sampling_weights"] = [1.0, 1.0]
    parameters["model"]["print_summary"] = True
    parameters["track_memory_usage"] = True
    parameters["verbose"] = True
    parameters = populate_header_in_parameters(parameters, parameters["headers"])

    dataset = ImagesFromDataFrame(
        training_data, parameters, train=True, loader_type="train"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=parameters["batch_size"], shuffle=True
    )
    parameters = populate_channel_keys_in_params(loader, parameters)

    module = GandlfLightningModule(parameters, output_dir=TESTS_DIRPATH)
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        fast_dev_run=False,
        devices=1,
        num_nodes=1,
        max_epochs=1,
        sync_batchnorm=True,
    )
    trainer.fit(module, loader)


# @pytest.mark.skipif(
#     torch.cuda.device_count() < 2, reason="Test requires at least 2 GPUs to run"
# )
@pytest.mark.skip(
    reason="This test is failing due to a torchio problems with distributed data parallel"
)
def test_port_model_forward_2d_rad_segmentation_multi_gpu_single_node(device):
    parameters = parseConfig(
        TESTS_DIRPATH + "/config_segmentation.yaml", version_check_flag=False
    )

    training_data, parameters["headers"] = parseTrainingCSV(
        TEST_DATA_DIRPATH + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = PATCH_SIZE["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
    parameters["class_weights"] = [1.0, 1.0]
    parameters["sampling_weights"] = [1.0, 1.0]
    parameters = populate_header_in_parameters(parameters, parameters["headers"])

    dataset = ImagesFromDataFrame(
        training_data, parameters, train=True, loader_type="train"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=parameters["batch_size"], shuffle=True
    )
    parameters = populate_channel_keys_in_params(loader, parameters)

    module = GandlfLightningModule(parameters, output_dir=TESTS_DIRPATH)
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        fast_dev_run=False,
        devices="auto",
        num_nodes=1,
        max_epochs=1,
        sync_batchnorm=True,
    )
    trainer.fit(module, loader)
