import yaml
from typing import Union
from .dicomanonymizer.dicomanonymizer import anonymize


def run_anonymizer(
    input_path: str, output_path: str, parameters: Union[str, list, int]
):
    """
    This function performs anonymization of a single image or a collection of images.

    Args:
        input_path (str): The input file or folder.
        output_path (str): The output file or folder.
        parameters (Union[str, list, int]): The parameters for anonymization; for DICOM scans, the only optional argument is "delete_private_tags", which defaults to True.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    if not isinstance(parameters, dict):
        with open(parameters, "r") as file_data:
            yaml_data = file_data.read()
        parameters = yaml.safe_load(yaml_data)

    if "rad" in parameters["modality"]:
        if "delete_private_tags" not in parameters:
            parameters["delete_private_tags"] = True
        return anonymize(
            input_path,
            output_path,
            anonymization_actions={},
            deletePrivateTags=parameters["delete_private_tags"],
        )
    elif parameters["modality"] in ["histo", "path"]:
        raise NotImplementedError(
            "Anonymization for histology images has not been implemented yet."
        )
