### Parameters Configuration
We use the Pydantic library for parameter configuration. Parameters are organized by context within the base model classes described below.

#### Basic Classes
- **DefaultParameters**: Contains parameters initialized directly from the application.
- **UserDefinedParameters**: Contains parameters that the user must define.
##### Other Subclasses
- **ModelConfig**: Contains parameters specific to the model.
- **OptimizerConfig**: Contains parameters for the optimizer.
- **SchedulerConfig**: Contains parameters for the scheduler.
- **NestedTrainingConfig**: Contains parameters for nested training.
- **PatchSampleConfig**: Contains parameters for the patch sampler.

#### How to Define New Parameters
To define new parameters, add new parameters directly in the classes. 
Also, create a new BaseModel class and add it to one of the basic classes (UserDefinedParameters or DefaultParameters).

If validation is required, you can define it in the validators file.
For more details, refer to the [Pydantic documentation](https://docs.pydantic.dev/latest/).
 