import sys
import torch


# Dice scores and dice losses
def dice(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    This function computes a dice score between two tensors.

    Args:
        predicted (torch.Tensor): Predicted value by the network.
        target (torch.Tensor): Required target label to match the predicted with

    Returns:
        torch.Tensor: The computed dice score.
    """
    predicted_flat = predicted.flatten()
    label_flat = target.flatten()
    intersection = (predicted_flat * label_flat).sum()

    dice_score = (2.0 * intersection + sys.float_info.min) / (
        predicted_flat.sum() + label_flat.sum() + sys.float_info.min
    )

    return dice_score


def mcc(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the Matthews Correlation Coefficient (MCC) between two tensors. Adapted from https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py.

    Args:
        predictions (torch.Tensor): The predicted value by the network.
        targets (torch.Tensor): Required target label to match the predicted with

    Returns:
        torch.Tensor: The computed MCC score.
    """
    tp = torch.sum(torch.mul(predictions, targets))
    tn = torch.sum(torch.mul((1 - predictions), (1 - targets)))
    fp = torch.sum(torch.mul(predictions, (1 - targets)))
    fn = torch.sum(torch.mul((1 - predictions), targets))

    numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
    # Adding epsilon to the denominator to avoid divide-by-zero errors.
    denominator = (
        torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )
        + torch.finfo(torch.float32).eps
    )

    return torch.div(numerator.sum(), denominator.sum())


def generic_loss_calculator(
    predicted: torch.Tensor,
    target: torch.Tensor,
    num_class: int,
    loss_criteria,
    weights: list = None,
    ignore_class: int = None,
    loss_type: int = 0,
) -> torch.Tensor:
    """
    This function computes the mean class dice score between two tensors

    Args:
        predicted (torch.Tensor): Predicted generally by the network
        target (torch.Tensor): Required target label to match the predicted with
        num_class (int): Number of classes (including the background class)
        loss_criteria (function): Loss function to use
        weights (list, optional): Dice weights for each class (excluding the background class), defaults to None
        ignore_class (int, optional): Class to ignore, defaults to None
        loss_type (int, optional): Type of loss to compute, defaults to 0. The options are:
            0: no loss, normal dice calculation
            1: dice loss, (1-dice)
            2: log dice, -log(dice)

    Returns:
        torch.Tensor: Mean Class Dice score
    """
    accumulated_loss = 0
    # default to a ridiculous value so that it is ignored by default
    ignore_class = -1e10 if ignore_class is None else ignore_class

    for class_index in range(num_class):
        if class_index != ignore_class:
            current_loss = loss_criteria(
                predicted[:, class_index, ...], target[:, class_index, ...]
            )

            # subtract from 1 because this is supposed to be a loss
            default_loss = 1 - current_loss
            if loss_type == 2 or loss_type == "log":
                # negative because we want positive losses, and add epsilon to avoid infinities
                current_loss = -torch.log(current_loss + torch.finfo(torch.float32).eps)
            else:
                current_loss = default_loss

            # multiply by appropriate weight if provided
            if weights is not None:
                current_loss = current_loss * weights[class_index]

            accumulated_loss += current_loss

    if weights is None:
        accumulated_loss /= num_class

    return accumulated_loss


def MCD_loss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the Dice loss between two tensors. These weights should be the penalty weights, not dice weights.

    Args:
        predicted (torch.Tensor): The predicted value by the network.
        target (torch.Tensor): Required target label to match the predicted with
        params (dict): Dictionary of parameters

    Returns:
        torch.Tensor: The computed MCC loss.
    """
    return generic_loss_calculator(
        predicted,
        target,
        len(params["model"]["class_list"]),
        dice,
        params["weights"],
        None,
        1,
    )


def MCD_log_loss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the Dice loss between two tensors with log. These weights should be the penalty weights, not dice weights.

    Args:
        predicted (torch.Tensor): The predicted value by the network.
        target (torch.Tensor): Required target label to match the predicted with
        params (dict): Dictionary of parameters

    Returns:
        torch.Tensor: The computed MCC loss.
    """
    return generic_loss_calculator(
        predicted,
        target,
        len(params["model"]["class_list"]),
        dice,
        params["weights"],
        None,
        2,
    )


def MCC_loss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the Matthews Correlation Coefficient (MCC) loss between two tensors. These weights should be the penalty weights, not dice weights.

    Args:
        predicted (torch.Tensor): The predicted value by the network.
        target (torch.Tensor): Required target label to match the predicted with
        params (dict): Dictionary of parameters

    Returns:
        torch.Tensor: The computed MCC loss.
    """
    return generic_loss_calculator(
        predicted,
        target,
        len(params["model"]["class_list"]),
        mcc,
        params["weights"],
        None,
        1,
    )


def MCC_log_loss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the Matthews Correlation Coefficient (MCC) loss between two tensors with log. These weights should be the penalty weights, not dice weights.

    Args:
        predicted (torch.Tensor): The predicted value by the network.
        target (torch.Tensor): Required target label to match the predicted with
        params (dict): Dictionary of parameters

    Returns:
        torch.Tensor: The computed MCC loss.
    """
    return generic_loss_calculator(
        predicted,
        target,
        len(params["model"]["class_list"]),
        mcc,
        params["weights"],
        None,
        2,
    )


def tversky_loss(
    predicted: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5
) -> torch.Tensor:
    """
    This function calculates the Tversky loss between two tensors.

    Args:
        predicted (torch.Tensor): Predicted generally by the network
        target (torch.Tensor): Required target label to match the predicted with
        alpha (float, optional): Weight of false positives. Defaults to 0.5.
        beta (float, optional): Weight of false negatives. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed Tversky Loss
    """
    # Move this part later to parameter parsing, no need to check every time
    assert 0 <= alpha <= 1, f"Invalid alpha value: {alpha}"
    assert 0 <= beta <= 1, f"Invalid beta value: {beta}"
    assert 0 <= alpha + beta <= 1, f"Invalid alpha and beta values: {alpha}, {beta}"

    predicted_flat = predicted.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    true_positives = (predicted_flat * target_flat).sum()
    false_positives = ((1 - target_flat) * predicted_flat).sum()
    false_negatives = (target_flat * (1 - predicted_flat)).sum()

    numerator = true_positives
    denominator = true_positives + alpha * false_positives + beta * false_negatives
    score = (numerator + sys.float_info.min) / (denominator + sys.float_info.min)

    loss = 1 - score
    return loss


def MCT_loss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict = None
) -> torch.Tensor:
    """
    This function calculates the Multi-Class Tversky loss between two tensors.

    Args:
        predicted (torch.Tensor): Predicted generally by the network
        target (torch.Tensor): Required target label to match the predicted with
        params (dict, optional): Additional parameters for computing loss function, including weights for each class

    Returns:
        torch.Tensor: Computed Multi-Class Tversky Loss
    """

    acc_tv_loss = 0
    num_classes = predicted.shape[1]

    for i in range(num_classes):
        curr_loss = tversky_loss(predicted[:, i, ...], target[:, i, ...])
        if params is not None and params.get("weights") is not None:
            curr_loss = curr_loss * params["weights"][i]
        acc_tv_loss += curr_loss

    if params is not None and params.get("weights") is None:
        acc_tv_loss /= num_classes

    return acc_tv_loss


def KullbackLeiblerDivergence(mu, logvar, params=None):
    """
    Calculates the Kullback-Leibler divergence between two Gaussian distributions.

    Args:
        mu (torch.Tensor): The mean of the first Gaussian distribution
        logvar (torch.Tensor): The logarithm of the variance of the first Gaussian distribution
        params (dict, optional): A dictionary of optional parameters

    Returns:
        torch.Tensor: The computed Kullback-Leibler divergence
    """
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return loss.mean()


def FocalLoss(
    predicted: torch.Tensor, target: torch.Tensor, params: dict = None
) -> torch.Tensor:
    """
    This function calculates the Focal loss between two tensors.

    Args:
        predicted (torch.Tensor): Predicted generally by the network
        target (torch.Tensor): Required target label to match the predicted with
        params (dict, optional): Additional parameters for computing loss function, including weights for each class

    Returns:
        torch.Tensor: Computed Focal Loss
    """
    gamma = 2.0
    size_average = True
    if isinstance(params["loss_function"], dict):
        gamma = params["loss_function"].get("gamma", 2.0)
        size_average = params["loss_function"].get("size_average", True)

    def _focal_loss(preds, target, gamma, size_average=True):
        """
        Internal helper function to calculate focal loss for a single class.

        Args:
            preds (torch.Tensor): predicted generally by the network
            target (torch.Tensor): Required target label to match the predicted with
            gamma (float): The gamma value for focal loss
            size_average (bool, optional): Whether to average the loss across the batch. Defaults to True.

        Returns:
            torch.Tensor: Computed focal loss for a single class.
        """
        ce_loss = torch.nn.CrossEntropyLoss(reduce=False)
        logpt = ce_loss(preds, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** gamma) * logpt
        return_loss = loss.sum()
        if size_average:
            return_loss = loss.mean()
        return return_loss

    acc_focal_loss = 0
    num_classes = predicted.shape[1]

    for i in range(num_classes):
        curr_loss = _focal_loss(
            predicted[:, i, ...], target[:, i, ...], gamma, size_average
        )
        if params is not None and params.get("weights") is not None:
            curr_loss = curr_loss * params["weights"][i]
        acc_focal_loss += curr_loss

    return acc_focal_loss
