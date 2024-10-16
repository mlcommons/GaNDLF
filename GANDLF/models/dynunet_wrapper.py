from .modelBase import ModelBase
import monai.networks.nets.dynunet as dynunet


def get_kernels_strides(sizes, spacings):
    """
    More info: https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/create_network.py#L19

    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            assert (
                i % j == 0
            ), f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class dynunet_wrapper(ModelBase):
    """
    More info: https://docs.monai.io/en/stable/networks.html#dynunet

    Args:
        spatial_dims (int): number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (Sequence[Union[Sequence[int], int]]): convolution kernel size.
        strides (Sequence[Union[Sequence[int], int]]): convolution strides for each blocks.
        upsample_kernel_size (Sequence[Union[Sequence[int], int]]): convolution kernel size for transposed convolution layers. The values should equal to strides[1:].
        filters (Optional[Sequence[int]]): number of output channels for each blocks.  Defaults to None.
        dropout (Union[Tuple, str, float, None]): dropout ratio. Defaults to no dropout.
        norm_name (Union[Tuple, str]): feature normalization type and arguments. Defaults to INSTANCE.
        act_name (Union[Tuple, str]): activation layer type and arguments. Defaults to leakyrelu.
        deep_supervision (bool): whether to add deep supervision head before output. Defaults to False.
        deep_supr_num (int): number of feature maps that will output during deep supervision head. The value should be larger than 0 and less than the number of up sample layers. Defaults to 1.
        res_block (bool): whether to use residual connection based convolution blocks during the network. Defaults to False.
        trans_bias (bool): whether to set the bias parameter in transposed convolution layers. Defaults to False.
    """

    def __init__(self, parameters: dict):
        super(dynunet_wrapper, self).__init__(parameters)

        patch_size = parameters.get("patch_size", None)
        spacing = parameters.get(
            "spacing_for_internal_computations",
            [1.0 for i in range(parameters["model"]["dimension"])],
        )
        parameters["model"]["kernel_size"] = parameters["model"].get(
            "kernel_size", None
        )
        parameters["model"]["strides"] = parameters["model"].get("strides", None)
        if (parameters["model"]["kernel_size"] is None) or (
            parameters["model"]["strides"] is None
        ):
            kernel_size, strides = get_kernels_strides(patch_size, spacing)
            parameters["model"]["kernel_size"] = kernel_size
            parameters["model"]["strides"] = strides

        parameters["model"]["filters"] = parameters["model"].get("filters", None)
        parameters["model"]["act_name"] = parameters["model"].get(
            "act_name", ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        )
        parameters["model"]["deep_supervision"] = parameters["model"].get(
            "deep_supervision", False
        )
        parameters["model"]["deep_supr_num"] = parameters["model"].get(
            "deep_supr_num", 1
        )
        parameters["model"]["res_block"] = parameters["model"].get("res_block", True)
        parameters["model"]["trans_bias"] = parameters["model"].get("trans_bias", False)
        parameters["model"]["dropout"] = parameters["model"].get("dropout", None)

        if not ("norm_type" in parameters["model"]):
            self.norm_type = "INSTANCE"

        self.model = dynunet.DynUNet(
            spatial_dims=self.n_dimensions,
            in_channels=self.n_channels,
            out_channels=self.n_classes,
            kernel_size=parameters["model"]["kernel_size"],
            strides=parameters["model"]["strides"],
            upsample_kernel_size=parameters["model"]["strides"][1:],
            filters=parameters["model"][
                "filters"
            ],  # number of output channels for each blocks
            dropout=parameters["model"][
                "dropout"
            ],  # dropout ratio. Defaults to no dropout
            norm_name=self.norm_type,
            act_name=parameters["model"]["act_name"],
            deep_supervision=parameters["model"]["deep_supervision"],
            deep_supr_num=parameters["model"][
                "deep_supr_num"
            ],  # number of feature maps that will output during deep supervision head.
            res_block=parameters["model"]["res_block"],
            trans_bias=parameters["model"]["trans_bias"],
        )

    def forward(self, x):
        return self.model.forward(x)
