from .regression_new import BinaryCrossEntropyLoss, BinaryCrossEntropyWithLogitsLoss
from .segmentation_new import MulticlassDiceLoss, MulticlassFocalLoss
from .loss_interface import AbstractHybridLoss


class DiceCrossEntropyLoss(AbstractHybridLoss):
    def _initialize_all_loss_calculators(self):
        return [MulticlassDiceLoss(self.params), BinaryCrossEntropyLoss(self.params)]


class DiceCrossEntropyLossLogits(AbstractHybridLoss):
    def _initialize_all_loss_calculators(self):
        return [
            MulticlassDiceLoss(self.params),
            BinaryCrossEntropyWithLogitsLoss(self.params),
        ]


class DiceFocalLoss(AbstractHybridLoss):
    def _initialize_all_loss_calculators(self):
        return [MulticlassDiceLoss(self.params), MulticlassFocalLoss(self.params)]
