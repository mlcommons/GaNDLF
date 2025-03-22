from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Literal, Tuple
from GANDLF.optimizers import global_optimizer_dict

# takes the keys from global optimizer
OPTIMIZER_OPTIONS = Literal[tuple(global_optimizer_dict.keys())]


class sgd_config(BaseModel):
    momentum: float = Field(default=0.99)
    weight_decay: float = Field(default=3e-05)
    dampening: float = Field(default=0)
    nesterov: bool = Field(default=True)


class asgd_config(BaseModel):
    alpha: float = Field(default=0.75)
    t0: float = Field(default=1e6)
    lambd: float = Field(default=1e-4)
    weight_decay: float = Field(default=3e-05)


class adam_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.00005)
    eps: float = Field(default=1e-8)
    amsgrad: float = Field(default=False)


class adamax_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.00005)
    eps: float = Field(default=1e-8)


class rprop_config(BaseModel):
    etas: Tuple[float, float] = Field(default=(0.5, 1.2))
    step_sizes: Tuple[float, float] = Field(default=(1e-6, 50))


class adadelta_config(BaseModel):
    rho: float = Field(default=0.9)
    eps: float = Field(default=1e-6)
    weight_decay: float = Field(default=3e-05)


class adagrad_config(BaseModel):
    lr_decay: float = Field(default=0)
    eps: float = Field(default=1e-6)
    weight_decay: float = Field(default=3e-05)


class rmsprop_config(BaseModel):
    alpha: float = Field(default=0.99)
    eps: float = Field(default=1e-8)
    centered: bool = Field(default=False)
    momentum: float = Field(default=0)
    weight_decay: float = Field(default=3e-05)


class radam_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    foreach: bool = Field(default=None)


class nadam_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    foreach: bool = Field(default=None)


class novograd_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8)
    weight_decay: float = Field(default=3e-05)
    amsgrad: bool = Field(default=False)


class ademamix_config(BaseModel):
    pass  # TODO: Check it because the default parameters are not in the optimizer dict


class lion_config(BaseModel):
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    weight_decay: float = Field(default=0.0)
    decoupled_weight_decay: bool = Field(default=False)


class adopt_config(BaseModel):
    pass  # TODO: Check it because the default parameters are not in the optimizer dict


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: OPTIMIZER_OPTIONS = Field(description="Type of optimizer to use")


optimizer_dict_config = {
    "sgd": sgd_config,
    "asgd": asgd_config,
    "adam": adam_config,
    "adamw": adam_config,
    "adamax": adamax_config,
    # "sparseadam": sparseadam,
    "rprop": rprop_config,
    "adadelta": adadelta_config,
    "adagrad": adagrad_config,
    "rmsprop": rmsprop_config,
    "radam": radam_config,
    "novograd": novograd_config,
    "nadam": nadam_config,
    "ademamix": ademamix_config,
    "lion": lion_config,
    "adopt": adopt_config,
}
