import torch
from GANDLF.misc_utils.clip_gradients import dispatch_clip_grad_


class GradScaler:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        clip_mode="norm",
        parameters=None,
        create_graph=False,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(
                optimizer
            )  # unscale the gradients of optimizer's assigned params in-place
            if (clip_mode is None) or (str(clip_mode).lower() == "none"):
                clip_mode == "norm" # default, in case none gets passed
            dispatch_clip_grad_(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def model_parameters(model, exclude_head=False):
    if exclude_head:
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()
