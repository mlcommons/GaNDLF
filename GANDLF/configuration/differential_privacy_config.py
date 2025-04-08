from typing_extensions import Literal

from pydantic import BaseModel, Field, ConfigDict

ACCOUNTANT_OPTIONS = Literal["rdp", "gdp", "prv"]


class DifferentialPrivacyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    noise_multiplier: float = Field(default=10.0)
    max_grad_norm: float = Field(default=1.0)
    accountant: ACCOUNTANT_OPTIONS = Field(default="rdp")
    secure_mode: bool = Field(default=False)
    allow_opacus_model_fix: bool = Field(default=True)
    delta: float = Field(default=1e-5)
    physical_batch_size: int = Field(validate_default=True)
