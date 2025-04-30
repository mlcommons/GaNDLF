from pathlib import Path

import numpy as np

from panoptica import Panoptica_Evaluator


def generate_instance_segmentation(
    prediction: np.ndarray, target: np.ndarray, panoptica_config_path: str = None
) -> dict:
    """
    Evaluate a single exam using Panoptica.

    Args:
        prediction (np.ndarray): The input prediction containing objects.
        label_path (str): The path to the reference label.
        panoptica_config_path (str): The path to the Panoptica configuration file.

    Returns:
        dict: The evaluation results.
    """

    cwd = Path(__file__).parent.absolute()
    panoptica_config_path = (
        cwd / "panoptica_config_path.yaml"
        if panoptica_config_path is None
        else panoptica_config_path
    )
    evaluator = Panoptica_Evaluator.load_from_config(panoptica_config_path)

    # call evaluate
    group2result = evaluator.evaluate(prediction_arr=prediction, reference_arr=target)

    results = {k: r.to_dict() for k, r in group2result.items()}
    return results
