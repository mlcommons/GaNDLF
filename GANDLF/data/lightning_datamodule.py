import lightning.pytorch as pl
from GANDLF.compute.generic import (
    TrainingSubsetDataParser,
    ValidationSubsetDataParser,
    TestSubsetDataParser,
    InferenceSubsetDataParserRadiology,
)
from torch.utils.data import DataLoader as TorchDataLoader
from copy import deepcopy


class GandlfTrainingDatamodule(pl.LightningDataModule):
    def __init__(self, data_dict_files: dict, parameters_dict: dict):
        super().__init__()

        # Batch size here and reinitialization of dataloader parsers is used
        # in automatic batch size tuning

        self.batch_size = parameters_dict["batch_size"]

        # This init procedure is extreme hack, but the only way to get around the
        # need to modify the parameters dict during the parsing procedure

        params = deepcopy(parameters_dict)

        train_subset_parser = TrainingSubsetDataParser(
            data_dict_files["training"], params
        )
        self.training_dataset = train_subset_parser.create_subset_dataset()
        params = train_subset_parser.get_params_extended_with_subset_data()

        val_subset_parser = ValidationSubsetDataParser(
            data_dict_files["validation"], params
        )
        self.validation_dataset = val_subset_parser.create_subset_dataset()
        params = val_subset_parser.get_params_extended_with_subset_data()

        testing_data = data_dict_files.get("testing", None)
        self.test_dataset = None
        if testing_data is not None:
            test_subset_parser = TestSubsetDataParser(testing_data, params)
            self.test_dataset = test_subset_parser.create_subset_dataset()
            params = test_subset_parser.get_params_extended_with_subset_data()

        self.parameters_dict = params

    def _get_dataloader(self, dataset, batch_size: int, shuffle: bool):
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.updated_parameters_dict.get("num_workers_dataloader", 1),
            pin_memory=self.updated_parameters_dict.get("pin_memory_dataloader", False),
            prefetch_factor=self.updated_parameters_dict.get(
                "prefetch_factor_dataloader", 2
            ),
        )

    @property
    def updated_parameters_dict(self):
        return self.parameters_dict

    def train_dataloader(self):
        self.updated_parameters_dict["batch_size"] = self.batch_size
        return self._get_dataloader(
            self.training_dataset, self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._get_dataloader(
            self.validation_dataset, batch_size=1, shuffle=False
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return self._get_dataloader(self.test_dataset, batch_size=1, shuffle=False)


class GandlfInferenceDatamodule(pl.LightningDataModule):
    def __init__(self, dataframe, parameters_dict):
        super().__init__()
        self.dataframe = dataframe
        params = deepcopy(parameters_dict)
        self.parameters_dict = params
        if self.parameters_dict["modality"] == "rad":
            inference_subset_data_parser_radiology = InferenceSubsetDataParserRadiology(
                self.dataframe, params
            )
            self.inference_dataset = (
                inference_subset_data_parser_radiology.create_subset_dataset()
            )

            self.parameters_dict = (
                inference_subset_data_parser_radiology.get_params_extended_with_subset_data()
            )

    @property
    def updated_parameters_dict(self):
        return self.parameters_dict

    def predict_dataloader(self):
        if self.parameters_dict["modality"] == "rad":
            return TorchDataLoader(
                self.inference_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.updated_parameters_dict.get(
                    "num_workers_dataloader", 1
                ),
                pin_memory=self.updated_parameters_dict.get(
                    "pin_memory_dataloader", False
                ),
                prefetch_factor=self.updated_parameters_dict.get(
                    "prefetch_factor_dataloader", 2
                ),
            )
        elif self.parameters_dict["modality"] in ["path", "histo"]:
            return self.dataframe.iterrows()
