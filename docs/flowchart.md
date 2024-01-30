## Internal Connections of GaNDLF

### I/O and Top-Level Parsing
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

    subgraph Data Processing
        df --> |Training| TrainingManager[/TrainingManager/] --> Data_Training[(Data_Training)] --> training_loop[\training_loop\]
        df --> |Inference| InferenceManager[/InferenceManager/] --> Data_Inference[(Data_Inference)] --> inference_loop[\inference_loop\]
        parameters([parameters]) --> TrainingManager
        parameters --> InferenceManager
        training_loop -->|actual training including backpropagation| Training[\Training\]
        training_loop -->|validate model performance without backpropagation| Validation[\Validation\]
    end
```

### Creating PyTorch Compute Objects
```mermaid
flowchart 
    subgraph Object Creation
        training_loop[\training_loop\] --> create_pytorch_objects[\compute.generic.create_pytorch_objects\]
        inference_loop[\inference_loop\] --> create_pytorch_objects
        parameters([parameters]) <-->|updated| create_pytorch_objects
        create_pytorch_objects --> Data_Training[(Data_Training)]
        create_pytorch_objects --> Data_Validation[(Data_Validation)] 
        create_pytorch_objects --> model[[model]]
        create_pytorch_objects --> optimizer[[optimizer]]
        create_pytorch_objects --> scheduler[[scheduler]]
        create_pytorch_objects --> weights[[weights]]
    end
```

### Training
```mermaid
flowchart TD
    subgraph Training
        Data_Training[(Data_Training)] --> train_network[\train_network\]
        parameters([parameters]) --> train_network
        model[[model]] --> train_network
        optimizer[[optimizer]] --> train_network
        train_network -->|training mode| step[\compute.step\]
        step -->|loss backpropagation| optimizer
        step -->|latest model| save_model[\utils.modelio.save_model\]
        step -->|loss and metrics| log_metrics[[training_logger]]
    end

```

### Validation

```mermaid
flowchart TD
    subgraph Validation
        DataLoader_Validation[(DataLoader_Validation)] --> validate_network[\validate_network\]
        parameters([parameters]) --> validate_network
        model[[model]] --> validate_network
        optimizer[[optimizer]] --> validate_network
        validate_network -->|validate mode| step[\compute.step\]
        validate_network -->|if validation loss improves| save_model[\utils.modelio.save_model\]
        step -->|loss and metrics| log_metrics[[validation_logger]]
    end
```


### Inference
```mermaid
flowchart TD
    subgraph Inference
        model[[model]] --> inference_loop[\inference_loop\]
        inference_loop --> create_pytorch_objects[\compute.generic.create_pytorch_objects\]
        DataLoader_Inference[(DataLoader_Inference)] --> inference_loop
        inference_loop -->|inference mode| step[\compute.step\]
    end
```

### The Actual `compute.step` Routine

```mermaid
flowchart TD
    subgraph Step Function
        Each_Sample[(Each_Sample)] -->|forward pass| model[[model]]
        model[[model]] --> type{Compute Type}
        type -->|training/validation| loss(Return Loss, Metrics, & Prediction)
        type -->|inference| inferences_metrics(Return Only Prediction because no GT)
    end
```
