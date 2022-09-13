from .morphology import torch_morphological, fill_holes, cca

from .tensor import get_mapped_label


global_postprocessing_dict = {
    "fill_holes": fill_holes,
    "mapping": get_mapped_label,
    "morphology": torch_morphological,
    "cca": cca,
}

# append post_processing functions that are to be be applied after reverse one-hot encoding
postprocessing_after_reverse_one_hot_encoding = ["mapping"]
