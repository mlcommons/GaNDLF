"""Regression test: EfficientNet _MBConv6 depthwise conv must set groups (not dense)."""

# Drop into GaNDLF's testing suite. Fails on master (where depthconv1 is a dense conv with
# groups=1) and passes once groups=6 * num_in_feats is added to _MBConv6.depthconv1.

import torch.nn as nn

from GANDLF.models.efficientnet import _MBConv1, _MBConv6

_COMMON = {
    "kernel_size": 5,
    "stride": 1,
    "output_size": [14, 14],
    "Norm": nn.BatchNorm2d,
    "Conv": nn.Conv2d,
    "Pool": nn.AdaptiveAvgPool2d,
    "reduction": 4,
}


def test_efficientnet_block_is_depthwise():
    """_MBConv6's depthconv1 must be depthwise: groups == its channel count."""
    block = _MBConv6(num_in_feats=80, num_out_feats=112, **_COMMON)
    dw = block.depthconv1
    expected_channels = 6 * 80  # MBConv6 expands input by 6x before the depthwise conv
    assert dw.groups == expected_channels, (
        f"depthconv1 should be depthwise (groups={expected_channels}), "
        f"got groups={dw.groups} -- this is a dense conv, not depthwise."
    )


def test_efficientnet_block_param_budget():
    """A single deep-stage block should be ~0.2M params, not ~6M (the dense-conv bug)."""
    block = _MBConv6(num_in_feats=80, num_out_feats=112, **_COMMON)
    params = sum(p.numel() for p in block.parameters())
    assert params < 1_000_000, (
        f"_MBConv6 block has {params:,} params; expected < 1M. A value near 6M means "
        f"depthconv1 fell back to a dense convolution (missing groups=)."
    )


def test_efficientnet_blocks_are_consistent():
    """Both block types should use depthwise convolutions."""
    b1 = _MBConv1(num_in_feats=32, num_out_feats=16, **_COMMON)
    b6 = _MBConv6(num_in_feats=80, num_out_feats=112, **_COMMON)
    assert b1.depthconv1.groups == b1.depthconv1.in_channels
    assert b6.depthconv1.groups == b6.depthconv1.in_channels
