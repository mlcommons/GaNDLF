from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Any


class PostProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    fill_holes: Any = Field(default=None)
    mapping: dict = Field(default=None)
    morphology: Any = Field(default=None)
    cca: Any = Field(default=None)
