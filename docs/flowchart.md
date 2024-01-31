## Internal Connections of GaNDLF

- The flowcharts have been created using the [mermaid](https://mermaid-js.github.io/mermaid/#/) library. Documentation for the flowchart section can be found [here](https://mermaid.js.org/syntax/flowchart.html).

### I/O
```mermaid
flowchart TD
    subgraph Input Files
        InputData[(InputData)] 
        InputYAMLConfig 
    end

    subgraph Parsing
        InputData[(InputData)] -->|pandas| df[(DataFrame)]
        InputYAMLConfig --> ConfigManager[/ConfigManager/] -->|error checks and defaults| parameters([parameters])
    end
```

### Top-Level Parsing
```mermaid
flowchart TD
    df[(DataFrame)] --> |Training| TrainingManager[/TrainingManager/] --> Data_Training_And_Validation[(Data_Training_And_Validation)] --> training_loop[\training_loop\]
    df --> |Inference or Testing| InferenceManager[/InferenceManager/] --> Data_Testing[(Data_Testing)] --> inference_loop[\inference_loop\]
    parameters([parameters]) --> TrainingManager
    parameters --> InferenceManager
    training_loop -->|actual training including backpropagation| Training[\Training\]
    training_loop -->|validate model performance without backpropagation| Validation[\Validation\]
    training_loop <==> create_pytorch_objects[\compute.generic.create_pytorch_objects\]
    Data_Training_And_Validation --> create_pytorch_objects
    parameters --> create_pytorch_objects
    Data_Testing --> create_pytorch_objects
    inference_loop -->|only forward pass| Inference[\Inference\]
    inference_loop <==> create_pytorch_objects
    Training --> model[(trained model & associated metrics)]
    Validation --> model
    Inference --> Predictions[(Predictions)]
```

### Creating PyTorch Compute Objects using `GANDLF.compute.generic.create_pytorch_objects`

```mermaid
flowchart LR
    Data_Full[(Data_Full)] --> create_pytorch_objects[\compute.generic.create_pytorch_objects\]
    parameters([parameters]) <==>|updated| create_pytorch_objects
    create_pytorch_objects -->|cross-validation| Data_Training[(Data_Training)]
    create_pytorch_objects -->|cross-validation| Data_Validation[(Data_Validation)] 
    create_pytorch_objects -->|cross-validation or inference| Data_Testing[(Data_Testing)]
    create_pytorch_objects -->|weights are either initialized or loaded| model[[model]]
    create_pytorch_objects --> optimizer[[optimizer]]
    create_pytorch_objects --> scheduler[[scheduler]]
```

### Training
```mermaid
flowchart TD
    subgraph Training
        parameters([parameters]) --> train_network[\compute.training_loop.train_network\]
        train_network -->|Create Compute Objects| create_pytorch_objects[\compute.generic.create_pytorch_objects\]
        Data_Training[(Data_Training)] --> step[\compute.step\]
        model[[model]] --> step
        optimizer[[optimizer]] --> step
        step -->|loss backpropagation| optimizer
        create_pytorch_objects --> optimizer[[optimizer]]
        create_pytorch_objects --> scheduler[[scheduler]]
        scheduler -->|update learning rate| optimizer
        step -->|latest model| save_model[\utils.modelio.save_model\]
        step -->|loss and metrics| log_metrics[[training_logger]]
        
        parameters([parameters]) <==>|updated| create_pytorch_objects
        create_pytorch_objects --> Data_Training[(Data_Training)]
        create_pytorch_objects --> model[[model]]
    end
     
    subgraph Validation and Testing
        create_pytorch_objects --> Data_Validation[(Data_Validation)] -->|validation mode| validate_network[\validate_network\]
        create_pytorch_objects --> Data_Testing[(Data_Testing)] -->|testing mode| validate_network[\validate_network\]
    end
```

### Validation

```mermaid
flowchart TD
    Data_Validation[(Data_Validation)] --> validate_network[\compute.forward_pass.validate_network\]
    model[[model]] --> validate_network
    optimizer[[optimizer]] --> validate_network
    parameters([parameters]) --> validate_network
    validate_network -->|validation mode| step[\compute.step\]
    validate_network -->|if validation loss improves| save_model[\utils.modelio.save_model\]
    step -->|loss and metrics| log_metrics[[validation_logger]]
```



### Inference

```mermaid
flowchart TD
    Data_Testing[(Data_Testing)] -->|testing mode| validate_network[\compute.forward_pass.validate_network\]
    model[[model]] -->|testing mode| validate_network
    parameters([parameters]) --> validate_network
    validate_network -->|testing mode| step[\compute.step\]
        step --> Predictions[(Predictions)]
```

### The Actual `compute.step` Routine

```mermaid
flowchart TD
    Each_Sample --> Input_GroundTruth[(Input_GroundTruth)] -->type{compute.loss_and_metric}
    Each_Sample[(Each Sample)] -->Input_DataPoint
    Input_DataPoint[(Input_DataPoint)] -->|forward pass| model[[model]]
    model --> Prediction[(Prediction)]
    Prediction --> type{compute.loss_and_metric}
    parameters([parameters]) --> type
    type -->|training/validation mode| loss(Return Loss, Metrics, & Prediction)
    type -->|testing mode| inferences_metrics(Return Only Prediction because no GT)
    Prediction --> inferences_metrics
```

### The `GANDLF.data` Submodule

```mermaid
flowchart TD
    parameters([parameters]) --> ImagesFromDataFrame
    parameters --> data_augmentation[[data.augmentation]]
    parameters --> data_processing[[data.pre/post_processing]]
    df[(DataFrame)] --> ImagesFromDataFrame[\data.ImagesFromDataFrame\]
    data_augmentation -->|Training| tioq[[patch_based_queue]]
    data_processing -->|Training| tioq
    data_processing -->|Not Training| tiosd[[non-patched_queue]]
    ImagesFromDataFrame -->|Training| tioq
    ImagesFromDataFrame -->|Not Training| tiosd
```
