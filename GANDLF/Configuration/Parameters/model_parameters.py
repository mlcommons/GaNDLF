from pydantic import BaseModel, model_validator, Field, AliasChoices, ConfigDict
from typing_extensions import Self, Literal, Optional
from typing import Union
from GANDLF.Configuration.Parameters.validators import (
    validate_class_list,
    validate_norm_type,
)
from GANDLF.models import global_models_dict

# Define model architecture options
ARCHITECTURE_OPTIONS = Literal[tuple(global_models_dict.keys())]
NORM_TYPE_OPTIONS = Literal["batch", "instance", "none"]


# You can define new parameters for model here. Please read the pydantic documentation.
# It allows extra fields in model dict.
class Model(BaseModel):
    model_config = ConfigDict(
        extra="allow"
    )  #  it allows extra fields in the model dict
    dimension: Optional[int] = Field(description="Dimension.")
    architecture: Union[ARCHITECTURE_OPTIONS, dict] = Field(description="Architecture.")
    final_layer: str = Field(description="Final layer.")
    norm_type: Optional[NORM_TYPE_OPTIONS] = Field(
        description="Normalization type.", default="batch"
    )  # TODO: check it again
    base_filters: Optional[int] = Field(
        description="Base filters.", default=None, validate_default=True
    )  # default is 32
    class_list: Union[list, str] = Field(default=[], description="Class list.")
    num_channels: Optional[int] = Field(
        description="Number of channels.",
        validation_alias=AliasChoices(
            "num_channels", "n_channels", "channels", "model_channels"
        ),
        default=3,
    )  # TODO: check it
    type: Optional[str] = Field(description="Type of model.", default="torch")
    data_type: str = Field(description="Data type.", default="FP32")
    save_at_every_epoch: bool = Field(default=False, description="Save at every epoch.")
    amp: bool = Field(default=False, description="Amplifier.")
    ignore_label_validation: Union[int, None] = Field(
        default=None, description="Ignore label validation."
    )  # TODO:  To check it
    print_summary: bool = Field(default=True, description="Print summary.")

    @model_validator(mode="after")
    def model_validate(self) -> Self:
        # TODO: Change the print to logging.warnings
        self.class_list = validate_class_list(
            self.class_list
        )  # init and validate the class_list parameter
        self.norm_type = validate_norm_type(
            self.norm_type, self.architecture
        )  # init and validate the norm type
        if self.amp is False:
            print("NOT using Mixed Precision Training")

        if self.save_at_every_epoch:
            print(
                "WARNING: 'save_at_every_epoch' will result in TREMENDOUS storage usage; use at your own risk."
            )  # TODO: It is better to use logging.warning

        if self.base_filters is None:
            self.base_filters = 32
            print("Using default 'base_filters' in 'model': ", self.base_filters)

        return self
