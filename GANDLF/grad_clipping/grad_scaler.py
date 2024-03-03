from typing import Optional
import torch
from GANDLF.grad_clipping.clip_gradients import dispatch_clip_grad_


class GradScaler:
    def __init__(self):
        """Initializes a GradScaler object with a PyTorch GradScaler."""
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        clip_mode: Optional[str] = "norm",
        parameters: Optional[torch.Tensor] = None,
        create_graph: Optional[bool] = False,
    ):
        """
        Scales the loss and performs backward pass through the computation graph.

        Args:
            loss (torch.Tensor): The loss tensor to scale and backpropagate.
            optimizer (torch.optim.Optimizer): The optimizer to step after backpropagation.
            clip_grad (Optional[float], optional): The clipping value/factor/norm, mode dependent. Defaults to None.
            clip_mode (Optional[str], optional): The mode of clipping. Defaults to "norm".
            parameters (Optional[torch.Tensor], optional): The model parameters to be clipped. Defaults to None.
            create_graph (Optional[bool], optional): Whether to create the graph. Defaults to False.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            # unscale the gradients of optimizer's assigned params in-place
            self._scaler.unscale_(optimizer)
            if (clip_mode is None) or (str(clip_mode).lower() == "none"):
                clip_mode = "norm"  # default, in case none gets passed
            dispatch_clip_grad_(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        """
        Returns the state dict of the underlying GradScaler.
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state dict of the underlying GradScaler.

        Args:
            state_dict (dict): The state dict to load.
        """
        self._scaler.load_state_dict(state_dict)


def model_parameters_exclude_head(
    model: torch.nn.Module, clip_mode: Optional[str] = None
):
    """
    Returns the parameters of a PyTorch model excluding the last two layers (the head).

    Args:
        model (torch.nn.Module): The PyTorch model to get the parameters from.
        clip_mode (Optional[str], optional): The mode of clipping. Defaults to None.

    Returns:
        Iterable: The model parameters excluding the last two layers if clip_mode is 'agc', otherwise all parameters.
    """
    exclude_head = False
    clip_mode = str(clip_mode).lower() if clip_mode is not None else None
    if clip_mode == "agc":
        exclude_head = True
    if exclude_head:
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()
