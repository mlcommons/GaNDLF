from pathlib import Path
import tempfile

import numpy as np

from panoptica import Panoptica_Evaluator


def generate_instance_segmentation(
    prediction: np.ndarray,
    target: np.ndarray,
    parameters: dict = None,
    panoptica_config_path: str = None,
) -> dict:
    """
    Evaluate a single exam using Panoptica.

    Args:
        prediction (np.ndarray): The input prediction containing objects.
        label_path (str): The path to the reference label.
        target (np.ndarray): The input target containing objects.
        panoptica_config_path (str): The path to the Panoptica configuration file.

    Returns:
        dict: The evaluation results.
    """

    cwd = Path(__file__).parent.absolute()
    # the parameters dict takes precedence over the panoptica_config_path
    panoptica_config = parameters.get("panoptica_config", None)
    if panoptica_config is None:
        panoptica_config_path = (
            cwd / "panoptica_config_brats.yaml"
            if panoptica_config_path is None
            else panoptica_config_path
        )
    else:
        # write the panoptica config to a file
        panoptica_config_path = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ).name
        with open(panoptica_config_path, "w") as f:
            f.write(panoptica_config)
    evaluator = Panoptica_Evaluator.load_from_config(panoptica_config_path)

    # call evaluate
    group2result = evaluator.evaluate(prediction_arr=prediction, reference_arr=target)

    results = {k: r.to_dict() for k, r in group2result.items()}
    return results
