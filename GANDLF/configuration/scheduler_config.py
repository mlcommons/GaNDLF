from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal
from typing import Union
from GANDLF.schedulers import global_schedulers_dict

TYPE_OPTIONS = Literal[tuple(global_schedulers_dict.keys())]


class base_triangle_config(BaseModel):
    min_lr: float = Field(default=(10**-3))
    max_lr: float = Field(default=1)
    step_size: float = Field(description="step_size", default=None)


class triangle_modified_config(BaseModel):
    min_lr: float = Field(default=0.000001)
    max_lr: float = Field(default=0.001)
    max_lr_multiplier: float = Field(default=1.0)
    step_size: float = Field(description="step_size", default=None)


class cyclic_lr_base_config(BaseModel):
    # More details https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
    min_lr: float = Field(
        default=None
    )  # The default value is calculated according the learning rate  * 0.001
    max_lr: float = Field(default=None)  # calculate in the validation stage
    gamma: float = Field(default=0.1)
    scale_mode: Literal["cycle", "iterations"] = Field(default="cycle")
    cycle_momentum: bool = Field(default=False)
    base_momentum: float = Field(default=0.8)
    max_momentum: float = Field(default=0.9)
    step_size: float = Field(description="step_size", default=None)


class exp_config(BaseModel):
    gamma: float = Field(default=0.1)


class step_config(BaseModel):
    gamma: float = Field(default=0.1)
    step_size: float = Field(description="step_size", default=None)


class cosineannealing_config(BaseModel):
    # More details https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    T_0: int = Field(default=5)
    T_mult: float = Field(default=1)
    min_lr: float = Field(default=0.001)


class reduce_on_plateau_config(BaseModel):
    # More details https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    min_lr: Union[float, list] = Field(default=None)
    gamma: float = Field(default=0.1)
    mode: Literal["min", "max"] = Field(default="min")
    factor: float = Field(default=0.1)
    patience: int = Field(default=10)
    threshold: float = Field(default=0.0001)
    cooldown: int = Field(default=0)
    threshold_mode: Literal["rel", "abs"] = Field(default="rel")


class warmupcosineschedule_config(BaseModel):
    # More details https://docs.monai.io/en/stable/optimizers.html#monai.optimizers.WarmupCosineSchedule
    warmup_steps: int = Field(default=None)


# It allows extra parameters
class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: TYPE_OPTIONS = Field(description="scheduler type")


# Define the type and the scheduler base model class
schedulers_dict_config = {
    "triangle": base_triangle_config,
    "triangle_modified": triangle_modified_config,
    "triangular": cyclic_lr_base_config,
    "exp_range": cyclic_lr_base_config,
    "exp": exp_config,
    "exponential": exp_config,
    "step": step_config,
    "reduce_on_plateau": reduce_on_plateau_config,
    "reduce-on-plateau": reduce_on_plateau_config,
    "plateau": reduce_on_plateau_config,
    "reduceonplateau": reduce_on_plateau_config,
    "cosineannealing": cosineannealing_config,
    "warmupcosineschedule": warmupcosineschedule_config,
    "wcs": warmupcosineschedule_config,
}
