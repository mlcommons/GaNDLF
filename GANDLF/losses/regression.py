from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from GANDLF.utils import one_hot


def CEL(
    prediction: torch.Tensor, target: torch.Tensor, params: dict = None
) -> torch.Tensor:
    """
    Cross entropy loss with optional class weights.

    Args:
        prediction (torch.Tensor): prediction tensor from the model.
        target (torch.Tensor): Target tensor of class targets.
        params (dict): Dictionary of parameters including weights.

    Returns:
        torch.Tensor: Cross entropy loss tensor.
    """
    if len(target.shape) > 1 and target.shape[-1] == 1:
        target = torch.squeeze(target, -1)

    weights = None
    if params.get("penalty_weights") is not None:
        # Check that the number of classes matches the number of weights
        num_classes = len(params["penalty_weights"])
        assert (
            prediction.shape[-1] == num_classes
        ), f"Number of classes {num_classes} does not match prediction shape {prediction.shape[-1]}"

        weights = torch.FloatTensor(list(params["penalty_weights"].values()))
        weights = weights.float().to(target.device)

    cel = CrossEntropyLoss(weight=weights)
    return cel(prediction, target)


def CE_Logits(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss with logits.

    Args:
        prediction (torch.Tensor): Prediction tensor from the model.
        target (torch.Tensor): Target tensor of binary targets.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.
    """
    assert torch.all(target.byte() == target), "Target tensor must be binary (0 or 1)"

    loss = torch.nn.BCEWithLogitsLoss()
    loss_val = loss(prediction.contiguous().view(-1), target.contiguous().view(-1))

    return loss_val


def CE(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss.

    Args:
        prediction (torch.Tensor): Prediction tensor from the model.
        target (torch.Tensor): Target tensor of binary targets.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.
    """
    assert torch.all(target.byte() == target), "Target tensor must be binary (0 or 1)"

    loss = torch.nn.BCELoss()
    loss_val = loss(
        prediction.contiguous().view(-1).float(), target.contiguous().view(-1).float()
    )

    return loss_val


def CCE_Generic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    params: dict,
    CCE_Type: Optional[torch.nn.modules.loss._Loss] = CrossEntropyLoss,
) -> torch.Tensor:
    """
    Generic function to calculate CCE loss

    Args:
        prediction (torch.tensor): The predicted output value for each pixel. dimension: [batch, class, x, y, z].
        target (torch.tensor): The ground truth target for each pixel. dimension: [batch, class, x, y, z] factorial_class_list.
        params (dict): The parameter dictionary.
        CCE_Type (torch.nn.modules.loss._Loss, optional): The type of cross entropy loss to use. Defaults to CrossEntropyLoss.

    Returns:
        torch.tensor: The final loss value after taking multiple classes into consideration
    """

    acc_ce_loss = 0
    target = one_hot(target, params["model"]["class_list"]).type(prediction.dtype)

    for i in range(0, len(params["model"]["class_list"])):
        curr_ce_loss = CCE_Type(prediction[:, i, ...], target[:, i, ...])
        if params["penalty_weights"] is not None:
            curr_ce_loss = curr_ce_loss * params["penalty_weights"][i]
        acc_ce_loss += curr_ce_loss

    # Take the mean of the loss if weights are not provided.
    if params["penalty_weights"] is None:
        acc_ce_loss = torch.mean(acc_ce_loss)

    return acc_ce_loss


def L1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: Optional[str] = "mean",
    scaling_factor: Optional[float] = 1,
) -> torch.Tensor:
    """
    Calculate the mean absolute error between the output variable from the network and the target.

    Args:
        prediction (torch.Tensor): The prediction generated usually by the network.
        target (torch.Tensor): The target for the corresponding Tensor for which the output was generated.
        reduction (Optional[str], optional): The reduction method for the loss. Defaults to 'mean'.
        scaling_factor (Optional[float], optional): The scaling factor to multiply the target with. Defaults to 1.

    Returns:
        torch.Tensor: The mean absolute error loss.
    """
    scaling_factor = torch.as_tensor(
        scaling_factor, dtype=target.dtype, device=target.device
    )
    target = target.float() * scaling_factor
    loss = F.l1_loss(prediction, target, reduction=reduction)
    return loss


def L1_loss(
    prediction: torch.Tensor, target: torch.Tensor, params: Optional[dict] = None
) -> torch.Tensor:
    """
    Computes the L1 loss between the prediction tensor and the target tensor.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
        params (Optional[dict], optional): The dictionary of parameters. Defaults to None.

    Returns:
        loss (torch.Tensor): The computed L1 loss.
    """
    acc_mse_loss = 0

    if prediction.shape[0] == 1:
        if params is not None:
            acc_mse_loss += L1(
                prediction,
                target,
                reduction=params["loss_function"]["l1"]["reduction"],
                scaling_factor=params["scaling_factor"],
            )
        else:
            acc_mse_loss += L1(prediction, target)

    # Compute the L1 loss
    else:
        if params is not None:
            for i in range(0, params["model"]["num_classes"]):
                acc_mse_loss += L1(
                    prediction[:, i, ...],
                    target[:, i, ...],
                    reduction=params["loss_function"]["mse"]["reduction"],
                    scaling_factor=params["scaling_factor"],
                )
        else:
            for i in range(0, prediction.shape[1]):
                acc_mse_loss += L1(prediction[:, i, ...], target[:, i, ...])

    # Normalize the loss by the number of classes
    if params is not None:
        acc_mse_loss /= params["model"]["num_classes"]
    else:
        acc_mse_loss /= prediction.shape[1]

    return acc_mse_loss


def MSE(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: Optional[str] = "mean",
    scaling_factor: Optional[float] = 1,
) -> torch.Tensor:
    """
    Compute the mean squared error loss for the prediction and target

    Args:
        prediction (torch.Tensor): The prediction generated usually by the network.
        target (torch.Tensor): The target for the corresponding Tensor for which the output was generated.
        reduction (Optional[str], optional): The reduction method for the loss. Defaults to 'mean'.
        scaling_factor (Optional[float], optional): The scaling factor to multiply the target with. Defaults to 1.

    Returns:
        torch.Tensor: The computed mean squared error loss.
    """
    scaling_factor = torch.as_tensor(scaling_factor, dtype=torch.float32)
    target = target.float() * scaling_factor
    loss = F.mse_loss(prediction, target, reduction=reduction)
    return loss


def MSE_loss(
    prediction: torch.Tensor, target: torch.Tensor, params: Optional[dict] = None
) -> torch.Tensor:
    """
    Compute the mean squared error loss for the prediction and target.

    Args:
        prediction (torch.Tensor): The prediction generated usually by the network.
        target (torch.Tensor): The target for the corresponding Tensor for which the output was generated.
        params (Optional[dict], optional): The dictionary of parameters. Defaults to None.

    Returns:
        torch.Tensor: The computed mean squared error loss.
    """
    reduction = "mean"
    scaling_factor = 1
    if params is not None and "loss_function" in params:
        mse_params = params["loss_function"].get("mse", {})
        reduction = mse_params.get("reduction", reduction)
        scaling_factor = mse_params.get("scaling_factor", scaling_factor)

    if prediction.shape[0] == 1:
        acc_mse_loss = MSE(
            prediction, target, reduction=reduction, scaling_factor=scaling_factor
        )
    else:
        acc_mse_loss = 0
        for i in range(prediction.shape[1]):
            acc_mse_loss += MSE(
                prediction[:, i, ...],
                target[:, i, ...],
                reduction=reduction,
                scaling_factor=scaling_factor,
            )
        acc_mse_loss /= prediction.shape[1]

    return acc_mse_loss
