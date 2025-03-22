from pydantic import BaseModel, Field
from typing_extensions import Literal

TYPE_OPTIONS = Literal["uniform", "label"]


class PatchSamplerConfig(BaseModel):
    type: TYPE_OPTIONS = Field(default="uniform")
    enable_padding: bool = Field(default=False)
    padding_mode: str = Field(default="symmetric")
    biased_sampling: bool = Field(default=False)
