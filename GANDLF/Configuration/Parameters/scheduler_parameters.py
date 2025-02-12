from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from GANDLF.schedulers import global_schedulers_dict

TYPE_OPTIONS = Literal[tuple(global_schedulers_dict.keys())]


# It allows extra parameters
class Scheduler(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: TYPE_OPTIONS = Field(
        description="triangle/triangle_modified use LambdaLR but triangular/triangular2/exp_range uses CyclicLR"
    )
    # min_lr: 0.00001, #TODO: this should be defined ??
    # max_lr: 1, #TODO: this should be defined ??
    step_size: float = Field(description="step_size", default=None)
