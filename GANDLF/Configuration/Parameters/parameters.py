from pydantic import BaseModel, ConfigDict
from GANDLF.Configuration.Parameters.user_defined_parameters import (
    UserDefinedParameters,
)


class ParametersConfiguration(BaseModel):
    model_config = ConfigDict(extra="allow")


class Parameters(ParametersConfiguration, UserDefinedParameters):
    pass
