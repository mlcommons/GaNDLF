from pydantic import BaseModel, Field


class PatchSampler(BaseModel):
    type: str = Field(default="uniform")
    enable_padding: bool = Field(default=False)
    padding_mode: str = Field(default="symmetric")
    biased_sampling: bool = Field(default=False)
