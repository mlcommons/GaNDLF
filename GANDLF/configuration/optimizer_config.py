from typing import Tuple

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Literal

from GANDLF.optimizers import global_optimizer_dict

# takes the keys from global optimizer
OPTIMIZER_OPTIONS = Literal[tuple(global_optimizer_dict.keys())]


class SgdConfig(BaseModel):
    momentum: float = Field(default=0.99)
    weight_decay: float = Field(default=3e-05)
    dampening: float = Field(default=0)
    nesterov: bool = Field(default=True)


class AsgdConfig(BaseModel):
    alpha: float = Field(default=0.75)
    t0: float = Field(default=1e6)
    lambd: float = Field(default=1e-4)
    weight_decay: float = Field(default=3e-05)


class AdamConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.00005)
    eps: float = Field(default=1e-8)
    amsgrad: bool = Field(default=False)


class AdamaxConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.00005)
    eps: float = Field(default=1e-8)


class RpropConfig(BaseModel):
    etas: Tuple[float, float] = Field(default=(0.5, 1.2))
    step_sizes: Tuple[float, float] = Field(default=(1e-6, 50))


class AdadeltaConfig(BaseModel):
    rho: float = Field(default=0.9)
    eps: float = Field(default=1e-6)
    weight_decay: float = Field(default=3e-05)


class AdagradConfig(BaseModel):
    lr_decay: float = Field(default=0)
    eps: float = Field(default=1e-6)
    weight_decay: float = Field(default=3e-05)


class RmspropConfig(BaseModel):
    alpha: float = Field(default=0.99)
    eps: float = Field(default=1e-8)
    centered: bool = Field(default=False)
    momentum: float = Field(default=0)
    weight_decay: float = Field(default=3e-05)


class RadamConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    foreach: bool = Field(default=None)


class NadamConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    foreach: bool = Field(default=None)


class NovogradConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    amsgrad: bool = Field(default=False)


class AdemamixConfig(BaseModel):
    pass


class LionConfig(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.0)
    decoupled_weight_decay: bool = Field(default=False)


class AdoptConfig(BaseModel):
    pass


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: OPTIMIZER_OPTIONS = Field(description="Type of optimizer to use")


optimizer_dict_config = {
    "sgd": SgdConfig,
    "asgd": AsgdConfig,
    "adam": AdamConfig,
    "adamw": AdamConfig,
    "adamax": AdamaxConfig,
    # "sparseadam": sparseadam,
    "rprop": RpropConfig,
    "adadelta": AdadeltaConfig,
    "adagrad": AdagradConfig,
    "rmsprop": RmspropConfig,
    "radam": RadamConfig,
    "novograd": NovogradConfig,
    "nadam": NadamConfig,
    "ademamix": AdemamixConfig,
    "lion": LionConfig,
    "adopt": AdoptConfig,
}
