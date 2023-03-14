import torch
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from GANDLF.utils import one_hot


def CEL(out, target, params):
    """
    Cross entropy loss with optional class weights.

    Args:
        out (torch.Tensor): Output tensor from the model.
        target (torch.Tensor): Target tensor of class labels.
        params (dict): Dictionary of parameters including weights.

    Returns:
        torch.Tensor: Cross entropy loss tensor.
    """
    if len(target.shape) > 1 and target.shape[-1] == 1:
        target = torch.squeeze(target, -1)

    weights = None
    if params.get("weights") is not None:
        # Check that the number of classes matches the number of weights
        num_classes = len(params["weights"])
        if out.shape[-1] != num_classes:
            raise ValueError(f"Number of classes {num_classes} does not match output shape {out.shape[-1]}")
        
        weights = torch.FloatTensor(list(params["weights"].values()))
        weights = weights.float().to(target.device)

    cel = CrossEntropyLoss(weight=weights)
    return cel(out, target)


def CE_Logits(out, target):
    """
    Binary cross entropy loss with logits.

    Args:
        out (torch.Tensor): Output tensor from the model.
        target (torch.Tensor): Target tensor of binary labels.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.
    """
    if not torch.all(target.byte() == target):
        raise ValueError("Target tensor must be binary (0 or 1)")

    loss = torch.nn.BCEWithLogitsLoss()
    loss_val = loss(out.contiguous().view(-1), target.contiguous().view(-1))
    return loss_val


def CE(out, target):
    """
    Binary cross entropy loss.

    Args:
        out (torch.Tensor): Output tensor from the model.
        target (torch.Tensor): Target tensor of binary labels.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.
    """
    if not torch.all(target.byte() == target):
        raise ValueError("Target tensor must be binary (0 or 1)")

    loss = torch.nn.BCELoss()
    loss_val = loss(out.contiguous().view(-1).float(), target.contiguous().view(-1).float())
    return loss_val


def CCE_Generic(out, target, params, CCE_Type):
    """
    Generic function to calculate CCE loss

    Args:
        out (torch.tensor): The predicted output value for each pixel. dimension: [batch, class, x, y, z].
        target (torch.tensor): The ground truth label for each pixel. dimension: [batch, class, x, y, z] factorial_class_list.
        params (dict): The parameter dictionary.
        CCE_Type (torch.nn): The CE loss function type.

    Returns:
        torch.tensor: The final loss value after taking multiple classes into consideration
    """

    acc_ce_loss = 0
    target = one_hot(target, params["model"]["class_list"]).type(out.dtype)

    for i in range(0, len(params["model"]["class_list"])):
        curr_ce_loss = CCE_Type(out[:, i, ...], target[:, i, ...])
        if params["weights"] is not None:
            curr_ce_loss = curr_ce_loss * params["weights"][i]
        acc_ce_loss += curr_ce_loss

    # Take the mean of the loss if weights are not provided.
    if params["weights"] is None:
        total_loss = torch.mean(total_loss)

    return acc_ce_loss


def L1(output, label, reduction="mean", scaling_factor=1):
    """
    Calculate the mean absolute error between the output variable from the network and the target
    Parameters
    ----------
    output : torch.Tensor
        The output generated usually by the network
    label : torch.Tensor
        The label for the corresponding Tensor for which the output was generated
    reduction : str, optional
        The type of reduction to apply to the output. Can be "none", "mean", or "sum". Default is "mean".
    scaling_factor : int, optional
        The scaling factor to multiply the label with. Default is 1.
    Returns
    -------
    loss : torch.Tensor
        The computed Mean Absolute Error (L1) loss for the output and label
    """
    scaling_factor = torch.as_tensor(scaling_factor, dtype=label.dtype, device=label.device)
    label = label.float() * scaling_factor
    loss = F.l1_loss(output, label, reduction=reduction)
    return loss


def L1_loss(inp, target, params):
    """
    Computes the L1 loss between the input tensor and the target tensor.
    
    Parameters:
        inp (torch.Tensor): The input tensor.
        target (torch.Tensor): The target tensor.
        params (dict, optional): A dictionary of hyperparameters. Defaults to None.
        
    Returns:
        loss (torch.Tensor): The computed L1 loss.
    """
    loss = 0
    
    # Check if the input and target shapes are consistent
    if inp.shape != target.shape:
        raise ValueError("Input and target shapes are inconsistent.")
    
    # Compute the L1 loss
    for i in range(inp.shape[0]):
        if params is not None:
            loss += L1(inp[:, i, ...], target[:, i, ...],
                       reduction=params["loss_function"]["l1"]["reduction"],
                       scaling_factor=params["scaling_factor"])
        else:
            loss += L1(inp[:, i, ...], target[:, i, ...])
    
    # Normalize the loss by the number of classes
    if params is not None:
        loss /= inp.shape[1]

    return loss


def MSE(output, label, reduction="mean", scaling_factor=1):
    """
    Calculate the mean square error between the output variable from the network and the target
    Parameters
    ----------
    output : torch.Tensor
        The output generated usually by the network
    label : torch.Tensor
        The label for the corresponding Tensor for which the output was generated
    reduction : string, optional
        DESCRIPTION. The default is 'mean'.
    scaling_factor : float, optional
        The scaling factor to multiply the label with
    Returns
    -------
    loss : torch.Tensor
        Computed Mean Squared Error loss for the output and label
    """
    scaling_factor = torch.as_tensor(scaling_factor, dtype=torch.float32)
    label = label.float() * scaling_factor
    loss = F.mse_loss(output, label, reduction=reduction)
    return loss


def MSE_loss(inp, target, params=None):
    """
    Compute the mean squared error loss for the input and target
    
    Parameters
    ----------
    inp : torch.Tensor
        The input tensor
    target : torch.Tensor
        The target tensor
    params : dict, optional
        A dictionary of parameters. Default: None.
        If params is not None and contains the key "loss_function", the value of 
        "loss_function" is expected to be a dictionary with a key "mse", which 
        can contain the key "reduction" and/or "scaling_factor". If "reduction" is 
        not specified, the default is 'mean'. If "scaling_factor" is not specified,
        the default is 1.
    
    Returns
    -------
    acc_mse_loss : torch.Tensor
        Computed mean squared error loss for the input and target
    """
    reduction = "mean"
    scaling_factor = 1
    if params is not None and "loss_function" in params:
        mse_params = params["loss_function"].get("mse", {})
        reduction = mse_params.get("reduction", "mean")
        scaling_factor = mse_params.get("scaling_factor", 1)

    if inp.shape[0] == 1:
        acc_mse_loss = MSE(inp, target, reduction=reduction, scaling_factor=scaling_factor)
    else:
        acc_mse_loss = 0
        for i in range(inp.shape[1]):
            acc_mse_loss += MSE(inp[:, i, ...], target[:, i, ...], reduction=reduction, scaling_factor=scaling_factor)
        acc_mse_loss /= inp.shape[1]

    return acc_mse_loss
