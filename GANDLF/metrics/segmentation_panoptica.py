from pathlib import Path
import os

from typing import Optional
import numpy as np

from panoptica import Panoptica_Evaluator


def generate_instance_segmentation(
    prediction: np.ndarray,
    target: np.ndarray,
    parameters: dict,
    panoptica_config_path: Optional[str] = None,
) -> dict:
    """
    Evaluate a single exam using Panoptica.

    Args:
        prediction (np.ndarray): The input prediction containing objects.
        label_path (str): The path to the reference label.
        target (np.ndarray): The input target containing objects.
        parameters (dict): The GaNDLF parameters from which panoptica config is to be extracted.
        panoptica_config_path (str): The path to the Panoptica configuration file.

    Returns:
        dict: The evaluation results.
    """

    os.environ["PANOPTICA_CITATION_REMINDER"] = "False"

    # the parameters dict takes precedence over the panoptica_config_path
    evaluator = parameters.get("panoptica_config", None)
    if evaluator is None:
        cwd = Path(__file__).parent.absolute()
        panoptica_config_path = (
            str(cwd / "panoptica_config_brats.yaml")
            if panoptica_config_path is None
            else panoptica_config_path
        )
        evaluator = Panoptica_Evaluator.load_from_config(panoptica_config_path)

    assert evaluator is not None, "Panoptica evaluator could not be initialized."

    # call evaluate
    group2result = evaluator.evaluate(prediction_arr=prediction, reference_arr=target)

    results = {k: r.to_dict() for k, r in group2result.items()}
    return results
