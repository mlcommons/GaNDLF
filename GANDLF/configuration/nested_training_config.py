from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, Optional


class NestedTraining(BaseModel):
    stratified: bool = Field(
        default=False,
        description="this will perform stratified k-fold cross-validation but only with offline data splitting",
    )
    testing: int = Field(
        default=-5,
        description="this controls the number of testing data folds for final model evaluation; [NOT recommended] to disable this, use '1'",
        le=10,
    )
    validation: int = Field(
        default=-5,
        description="this controls the number of validation data folds to be used for model *selection* during training (not used for back-propagation)",
    )
    proportional: Optional[bool] = Field(default=False)

    @model_validator(mode="after")
    def validate_nested_training(self) -> Self:
        if self.proportional is not None:
            self.stratified = self.proportional
        return self
