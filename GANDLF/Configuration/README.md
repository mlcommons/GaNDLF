### Pydantic configuration
We use the pydantic library for parameters configuration.
We separate the parameters via their contex in the above base model classes.

The basic classes are:
- The **default_parameters** contains the parameters that are initialized from the application directly.
- The **UserDefinedParameters** contains the parameters that the user should define.

The other subclasses are:
- The ModelConfig contains the parameters for Model.
- The OptimizerConfig contains the parameters for the Optimizer.
- The SchedulerConfig contains the parameters for the Scheduler.
- The NestedTrainingConfig contains the parameters for the Nesting Training
- The PatchSampleConfig contains the parameters for the PatchSampler

### How to define new parameters.
You can define a new BaseModel class and add the parameter in one of the basic classes (UserDefineParameters,DefaultParameters). 
Also, in the validators file can be defined your validation if is needed. Please read the Pydantic documentation for more details.
