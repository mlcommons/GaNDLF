from pydantic import BaseModel, Field
from typing_extensions import Literal
from GANDLF.optimizers import global_optimizer_dict

# takes the keys from global optimizer
OPTIMIZER_OPTIONS = Literal[tuple(global_optimizer_dict.keys())]


class OptimizerConfig(BaseModel):
    type: OPTIMIZER_OPTIONS = Field(description="Type of optimizer to use")
