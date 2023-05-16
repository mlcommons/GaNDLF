"""
Exceptions for histology errors in GANDLF
"""

class DigitalPathologyError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class DigitalPathologyAugmentationError(DigitalPathologyError):
    """Error base class for all augmentation errors."""

    def __init__(self, *args):
        super().__init__(*args)


class InvalidRangeError(DigitalPathologyAugmentationError):
    """Raise when the range adjustment is not valid."""

    def __init__(self, title, range):
        super().__init__(f"Invalid range of {title}: {range}")
        self.range = range
        self.title = title


class TissueMaskException(Exception):
    pass
