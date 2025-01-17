import pandas as pd
import os, pickle, shutil
from pathlib import Path
from torch.profiler import profile, ProfilerActivity

from GANDLF.compute import training_loop
from GANDLF.utils import get_dataframe, split_data
from GANDLF.compute.generic import (
    TrainingSubsetDataParser,
    ValidationSubsetDataParser,
    TestSubsetDataParser,
)
import lightning.pytorch as pl
from warnings import warn
from GANDLF.models.lightning_module import GandlfLightningModule

import yaml


def TrainingManager(
    dataframe: pd.DataFrame, outputDir: str, parameters: dict, resume: bool, reset: bool
) -> None:
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe (pandas.DataFrame): The full data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
    """
    if reset:
        shutil.rmtree(outputDir)
        Path(outputDir).mkdir(parents=True, exist_ok=True)

    # save the current model configuration as a sanity check
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    dataframe_split = split_data(dataframe, parameters)

    last_indeces, _, _, _ = dataframe_split[-1]

    # check the last indeces to see if single fold training is requested
    singleFoldTesting = True if last_indeces[0] == 0 else False
    singleFoldValidation = True if last_indeces[1] == 0 else False

    for (
        testing_and_valid_indeces,
        trainingData,
        validationData,
        testingData,
    ) in dataframe_split:
        # the output of the current fold is only needed if multi-fold training is happening
        currentTestingOutputFolder = outputDir
        if not singleFoldTesting:
            currentTestingOutputFolder = os.path.join(
                outputDir, "testing_" + str(testing_and_valid_indeces[0])
            )
            Path(currentTestingOutputFolder).mkdir(parents=True, exist_ok=True)

        currentValidationOutputFolder = currentTestingOutputFolder
        if not singleFoldValidation:
            currentValidationOutputFolder = os.path.join(
                currentTestingOutputFolder, str(testing_and_valid_indeces[1])
            )
            Path(currentValidationOutputFolder).mkdir(parents=True, exist_ok=True)

        # initialize the dataframes and save them to disk
        data_dict = {
            "training": trainingData,
            "validation": validationData,
            "testing": testingData,
        }
        data_dict_files = {}
        for data_type, data in data_dict.items():
            data_dict_files[data_type] = None
            if data is not None:
                currentDataPickle = os.path.join(
                    currentValidationOutputFolder, "data_" + data_type + ".pkl"
                )
                data_dict_files[data_type] = currentDataPickle
                if (not os.path.exists(currentDataPickle)) or reset or resume:
                    data.to_pickle(currentDataPickle)
                    data.to_csv(currentDataPickle.replace(".pkl", ".csv"), index=False)
                else:
                    # read the data from the pickle if present
                    data_dict[data_type] = get_dataframe(currentDataPickle)

        # Dataloader initialization - should be extracted somewhere else (preferably abstracted away)

        train_subset_parser = TrainingSubsetDataParser(
            data_dict_files["training"], parameters
        )
        train_loader = train_subset_parser.create_subset_dataloader()
        parameters = train_subset_parser.get_params_extended_with_subset_data()

        val_subset_parser = ValidationSubsetDataParser(
            data_dict_files["validation"], parameters
        )
        val_loader = val_subset_parser.create_subset_dataloader()
        parameters = val_subset_parser.get_params_extended_with_subset_data()

        # This entire section should be handled in config parser

        accelerator = parameters.get("accelerator", "auto")
        allowed_accelerators = ["cpu", "gpu", "auto"]
        assert (
            accelerator in allowed_accelerators
        ), f"Invalid accelerator selected: {accelerator}. Please select from {allowed_accelerators}"
        strategy = parameters.get("strategy", "auto")
        allowed_strategies = ["auto", "ddp"]
        assert (
            strategy in allowed_strategies
        ), f"Invalid strategy selected: {strategy}. Please select from {allowed_strategies}"
        precision = parameters.get("precision", "32")
        allowed_precisions = [
            "64",
            "64-true",
            "32",
            "32-true",
            "16",
            "16-mixed",
            "bf16",
            "bf16-mixed",
        ]
        assert (
            precision in allowed_precisions
        ), f"Invalid precision selected: {precision}. Please select from {allowed_precisions}"

        warn(
            f"Using {accelerator} with {strategy} for training. Trainer will use only single accelerator instance. "
        )
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            fast_dev_run=False,
            devices=1,  # single-device-single-node forced now
            num_nodes=1,
            precision=precision,
            gradient_clip_algorithm=parameters["clip_mode"],
            gradient_clip_val=parameters["clip_grad"],
            max_epochs=parameters["num_epochs"],
            sync_batchnorm=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        module = GandlfLightningModule(
            parameters, output_dir=currentValidationOutputFolder
        )
        trainer.fit(module, train_loader, val_loader)

        testing_data = data_dict_files.get("testing", None)
        if testing_data:
            test_subset_parser = TestSubsetDataParser(
                data_dict_files["testing"], parameters
            )
            test_loader = test_subset_parser.create_subset_dataloader()
            parameters = test_subset_parser.get_params_extended_with_subset_data()
            trainer.test(module, test_loader)


def TrainingManager_split(
    dataframe_train: pd.DataFrame,
    dataframe_validation: pd.DataFrame,
    dataframe_testing: pd.DataFrame,
    outputDir: str,
    parameters: dict,
    resume: bool,
    reset: bool,
    _profile: bool,
):
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe_train (pd.DataFrame): The training data from CSV.
        dataframe_validation (pd.DataFrame): The validation data from CSV.
        dataframe_testing (pd.DataFrame): The testing data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
        _profile(bool):Whether the we want the profile activity or not.

    """
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    currentModelConfigYaml = os.path.join(outputDir, "config.yaml")

    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    if (not os.path.exists(currentModelConfigYaml)) or reset or resume:
        with open(currentModelConfigYaml, "w") as handle:
            yaml.dump(parameters, handle, default_flow_style=False)

    train_subset_parser = TrainingSubsetDataParser(dataframe_train, parameters)
    train_loader = train_subset_parser.create_subset_dataloader()
    parameters = train_subset_parser.get_params_extended_with_subset_data()

    val_subset_parser = ValidationSubsetDataParser(dataframe_validation, parameters)
    val_loader = val_subset_parser.create_subset_dataloader()
    parameters = val_subset_parser.get_params_extended_with_subset_data()

    # This entire section should be handled in config parser

    accelerator = parameters.get("accelerator", "auto")
    allowed_accelerators = ["cpu", "gpu", "auto"]
    assert (
        accelerator in allowed_accelerators
    ), f"Invalid accelerator selected: {accelerator}. Please select from {allowed_accelerators}"
    strategy = parameters.get("strategy", "auto")
    allowed_strategies = ["auto", "ddp"]
    assert (
        strategy in allowed_strategies
    ), f"Invalid strategy selected: {strategy}. Please select from {allowed_strategies}"
    precision = parameters.get("precision", "32")
    allowed_precisions = [
        "64",
        "64-true",
        "32",
        "32-true",
        "16",
        "16-mixed",
        "bf16",
        "bf16-mixed",
    ]
    assert (
        precision in allowed_precisions
    ), f"Invalid precision selected: {precision}. Please select from {allowed_precisions}"

    warn(
        f"Using {accelerator} with {strategy} for training. Trainer will use only single accelerator instance. "
    )
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        fast_dev_run=False,
        devices=1,  # single-device-single-node forced now
        num_nodes=1,
        precision=precision,
        gradient_clip_algorithm=parameters["clip_mode"],
        gradient_clip_val=parameters["clip_grad"],
        max_epochs=parameters["num_epochs"],
        sync_batchnorm=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    module = GandlfLightningModule(parameters, output_dir=outputDir)
    trainer.fit(module, train_loader, val_loader)

    if dataframe_testing:
        test_subset_parser = TestSubsetDataParser(dataframe_testing, parameters)
        test_loader = test_subset_parser.create_subset_dataloader()
        parameters = test_subset_parser.get_params_extended_with_subset_data()
        trainer.test(module, test_loader)
