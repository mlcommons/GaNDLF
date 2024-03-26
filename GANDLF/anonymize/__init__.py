import yaml
from typing import Union
from dicomanonymizer import anonymize
from .convert_to_nifti import convert_to_nifti

# from .slide_anonymizer import anonymize_slide


def run_anonymizer(
    input_path: str, output_path: str, parameters: Union[str, list, int], modality: str
) -> None:
    """
    This function performs anonymization of a single image or a collection of images.

    Args:
        input_path (str): The input file or folder.
        output_path (str): The output file or folder.
        parameters (Union[str, list, int]): The parameters for anonymization; for DICOM scans, the only optional argument is "delete_private_tags", which defaults to True.
        output_path (str): The modality type to process.
    """
    if parameters is None:
        parameters = {}
        parameters["modality"] = modality

    # read the parameters
    if not isinstance(parameters, dict):
        with open(parameters, "r") as file_data:
            yaml_data = file_data.read()
        parameters = yaml.safe_load(yaml_data)

    if "rad" in parameters["modality"]:
        # initialize defaults
        if "delete_private_tags" not in parameters:
            parameters["delete_private_tags"] = True
        if "convert_to_nifti" not in parameters:
            parameters["convert_to_nifti"] = False

        # if nifti conversion is requested, no other anonymization is required
        if parameters["convert_to_nifti"]:
            return convert_to_nifti(input_path, output_path)
        else:
            return anonymize(
                input_path,
                output_path,
                anonymization_actions={},
                delete_private_tags=parameters["delete_private_tags"],
            )
    elif parameters["modality"] in ["histo", "path"]:
        # anonymize_slide(
        #     input_path,
        #     output_path,
        # )
        raise NotImplementedError("Slide anonymization is not yet implemented.")
