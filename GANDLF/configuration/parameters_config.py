from pydantic import BaseModel, ConfigDict
from GANDLF.configuration.user_defined_config import UserDefinedParameters


class ParametersConfiguration(BaseModel):
    model_config = ConfigDict(extra="allow")


class Parameters(ParametersConfiguration, UserDefinedParameters):
    pass
