from .morphology import torch_morphological, fill_holes, cca

from .tensor import get_mapped_label


global_postprocessing_dict = {
    "fill_holes": fill_holes,
    "mapping": get_mapped_label,
    "cca": cca,
}
