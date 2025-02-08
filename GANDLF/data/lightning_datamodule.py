import torch
import lightning.pytorch as pl
from GANDLF.compute.generic import (
    TrainingSubsetDataParser,
    ValidationSubsetDataParser,
    TestSubsetDataParser,
)
from copy import deepcopy


class GandlfTrainingDatamodule(pl.LightningDataModule):
    def __init__(self, data_dict_files, parameters_dict):
        super().__init__()
        self.data_dict_files = data_dict_files

        # Batch size here and reinitialization of dataloader parsers is used
        # in automatic batch size tuning

        self.batch_size = parameters_dict["batch_size"]

        # This init procedure is extreme hack, but the only way to get around the
        # need to modify the parameters dict during the parsing procedure

        params = deepcopy(parameters_dict)

        self.train_subset_parser = TrainingSubsetDataParser(
            data_dict_files["training"], params
        )
        self.train_subset_parser.create_subset_dataloader()
        params = self.train_subset_parser.get_params_extended_with_subset_data()

        self.val_subset_parser = ValidationSubsetDataParser(
            data_dict_files["validation"], params
        )
        self.val_subset_parser.create_subset_dataloader()
        params = self.val_subset_parser.get_params_extended_with_subset_data()

        self.test_subset_parser = None
        testing_data = data_dict_files.get("testing", None)
        if testing_data:
            self.test_subset_parser = TestSubsetDataParser(
                data_dict_files["testing"], params
            )
            self.test_subset_parser.create_subset_dataloader()
            params = self.test_subset_parser.get_params_extended_with_subset_data()

        self.parameters_dict = params

    @property
    def updated_parameters_dict(self):
        return self.parameters_dict

    def train_dataloader(self):
        params = self.updated_parameters_dict
        params["batch_size"] = self.batch_size
        return TrainingSubsetDataParser(
            self.data_dict_files["training"], params
        ).create_subset_dataloader()

    def val_dataloader(self):
        params = self.updated_parameters_dict
        params["batch_size"] = self.batch_size
        return ValidationSubsetDataParser(
            self.data_dict_files["validation"], params
        ).create_subset_dataloader()

    def test_dataloader(self):
        if self.test_subset_parser is None:
            return None
        params = self.updated_parameters_dict
        params["batch_size"] = self.batch_size
        return TestSubsetDataParser(
            self.data_dict_files["testing"], params
        ).create_subset_dataloader()
