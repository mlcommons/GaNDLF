from typing import Any, ClassVar, List, Optional, Sequence, Union
from typing_extensions import Literal

import torch
from torch import Tensor
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.metric import Metric

from .functional import (
    lpips_compute,
    lpips_update,
    determine_converter,
    modify_net_input,
    modify_scaling_layer,
    _NoTrainLpipsLPIPSGandlf,
)


class LPIPSGandlf(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_scores: Tensor
    total: Tensor
    feature_network: str = "net"

    # due to the use of named tuple in the backbone the net variable cannot be scripted
    __jit_ignored_attributes__: ClassVar[List[str]] = ["net"]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        n_dim: int = 2,
        n_channels: int = 1,
        converter_type: Union[str, None] = None,
        **kwargs: Any,
    ):
        """Initialize the LPIPS metric for GanDLF. This metric is based on the
        torchmetrics implementation of LPIPS, with modifications to allow usage
        of single channel data and 3D data. Note that it uses the pre-trained
        model from the torchmetrics implementation, originally designed
        for 3-channel 2D data. Here the layers are modified, so results need
        to be interpreted with caution. For 2D 3-channel data, the results
        are expected to be similar to the original implementation.
        Args:
            net_type (Literal["vgg", "alex", "squeeze"]): The network type.
            reduction (Literal["sum", "mean"]): The reduction type, one of 'mean' or 'sum'
            normalize (bool): Whether to normalize the input images.
            n_dim (int): The number of dimensions of the input images.
            n_channels (int): The number of channels of the input images.
            converter_type (Union[str, None]): The converter type from ACS, one of
        'soft','asc' or 'conv3d'. If None, defaults to 'soft'.
            **kwargs: Additional arguments for the metric.
        """

        super().__init__(**kwargs)

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpipsLPIPSGandlf(n_dim=n_dim, net=net_type)

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
            )
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")
        if n_channels != 3:
            modify_scaling_layer(self.net)
            modify_net_input(self.net, net_type, n_channels)
        if n_dim == 3:
            modify_scaling_layer(self.net)
            converter = determine_converter(converter_type)
            converter(self.net)

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update internal states with lpips score."""
        loss, total = lpips_update(
            img1, img2, net=self.net, normalize=self.normalize
        )
        self.sum_scores += loss.sum()
        self.total += total

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        return lpips_compute(self.sum_scores, self.total, self.reduction)

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from GANDLF.metrics.gan_utils.lpip import LPIPSGandlf
            >>> metric = LPIPSGandlf(n_dim=2, n_channels=3, net_type='squeeze')
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from GANDLF.metrics.gan_utils.lpip import LPIPSGandlf
            >>> metric = LPIPSGandlf(n_dim=2, n_channels=3, net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


if __name__ == "__main__":
    lpip = LPIPSGandlf(
        n_dim=3,
        n_channels=3,
        net_type="squeeze",
        normalize=False,
    )
    rand_input_1 = (torch.rand(10, 3, 100, 100) * 2) - 1
    rand_input_2 = (torch.rand(10, 3, 100, 100) * 2) - 1
    # with torch.no_grad():
    print(lpip(rand_input_1, rand_input_2))

    import torchmetrics

    lpips_metric = (
        torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze",
            normalize=False,
            reduction="mean",
        )
    )
    print(lpips_metric(rand_input_1, rand_input_2))
