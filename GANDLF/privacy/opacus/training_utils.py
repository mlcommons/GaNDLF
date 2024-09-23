from typing import Tuple
import torch
import torchio


def handle_nonempty_batch(subject: dict, params: dict) -> Tuple[dict, int]:
    """
    Function to detect batch size from the subject an Opacus loader provides in the case of a non-empty batch, and make any changes to the subject dictionary that are needed for GaNDLF to use it.

    Args:
        subject (dict): Training data subject dictionary.
        params (dict): Training parameters.

    Returns:
        Tuple[dict, int]: Modified subject dictionary and batch size.
    """
    batch_size = len(subject[params["channel_keys"][0]][torchio.DATA])
    return subject, batch_size


def handle_empty_batch(subject: dict, params: dict, feature_shape: list) -> dict:
    """
    Function to replace the list of empty arrays an Opacus loader provides in the case of an empty batch with a subject dictionary GANDLF can consume.

    Args:
        subject (dict): Training data subject dictionary.
        params (dict): Training parameters.
        feature_shape (list): Shape of the features.

    Returns:
        dict: Modified subject dictionary.
    """

    print("\nConstructing empty batch dictionary.\n")

    subject = {
        "subject_id": "empty_batch",
        "spacing": None,
        "path_to_metadata": None,
        "location": None,
    }
    subject.update(
        {
            key: {torchio.DATA: torch.zeros(tuple([0] + feature_shape))}
            for key in params["channel_keys"]
        }
    )
    if params["problem_type"] != "segmentation":
        subject.update(
            {
                key: torch.zeros((0, params["model"]["num_classes"])).to(torch.int64)
                for key in params["value_keys"]
            }
        )
    else:
        subject.update(
            {
                "label": {
                    torchio.DATA: torch.zeros(tuple([0] + feature_shape)).to(
                        torch.int64
                    )
                }
            }
        )

    return subject


def handle_dynamic_batch_size(subject: dict, params: dict) -> Tuple[dict, int]:
    """
    Function to process the subject Opacus loaders provide and prepare to handle their dynamic batch size (including possible empty batches).

    Args:
        subject (dict): Training data subject dictionary.
        params (dict): Training parameters.

    Raises:
        RuntimeError: If the subject is a list object that is not an empty batch.

    Returns:
        Tuple[dict, int]: Modified subject dictionary and batch size.
    """

    # The handling performed here is currently to be able to comprehend what
    # batch size we are currently working with (which we may later see as not needed)
    # and also to handle the previously observed case where Opacus produces
    # a subject that is not a dictionary but rather a list of empty arrays
    # (due to the empty batch result). The latter case is detected as a subject that
    # is a list object.
    if isinstance(subject, list):
        are_empty = torch.Tensor(
            [torch.equal(tensor, torch.Tensor([])) for tensor in subject]
        )
        assert torch.all(
            are_empty
        ), "Detected a list subject that is not an empty batch, which is not expected behavior."
        # feature_shape = [params["model"]["num_channels"]]+params["patch_size"]
        feature_shape = [params["model"]["num_channels"]] + params["patch_size"]
        subject = handle_empty_batch(
            subject=subject, params=params, feature_shape=feature_shape
        )
        batch_size = 0
    else:
        subject, batch_size = handle_nonempty_batch(subject=subject, params=params)

    return subject, batch_size
