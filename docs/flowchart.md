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
        InputYAMLConfig --> ConfigManager[/ConfigManager/] -->|error checks and defaults| parameters
    end

    subgraph Data Processing
        df --> |Training| TrainingManager[/TrainingManager/] --> Data_Training[(Data_Training)] --> training_loop[\training_loop\]
        df --> |Inference| InferenceManager[/InferenceManager/] --> Data_Inference[(Data_Inference)] --> inference_loop[\inference_loop\]
        parameters --> TrainingManager
        parameters --> InferenceManager
    end
```

### Creating PyTorch Compute Objects
```mermaid
flowchart 
    subgraph Object Creation
        training_loop --> compute.generic.create_pytorch_objects
        parameters <-->|updated| compute.generic.create_pytorch_objects
        compute.generic.create_pytorch_objects --> model
        compute.generic.create_pytorch_objects --> optimizer
        compute.generic.create_pytorch_objects --> scheduler
        compute.generic.create_pytorch_objects --> DataLoader_Training
        compute.generic.create_pytorch_objects --> DataLoader_Validation
        compute.generic.create_pytorch_objects --> weights
    end
```

### Training
```mermaid
flowchart TD
    subgraph Training
        model --> train_network
        DataLoader_Training --> train_network
        optimizer --> train_network
        parameters --> train_network
        train_network -->|iterate subjects| compute.step.step
        compute.step.step -->|loss backpropagation| optimizer
        compute.step.step -->|latest model| utils.modelio.save_mode
    end

```

### validation
```mermaid
flowchart TD
    subgraph Validation
        model --> validate_network
        DataLoader_Validation --> validate_network
        optimizer --> validate_network
        validate_network -->|if validation loss improves| utils.modelio.save_mode
    end
```


### Interence
```mermaid
flowchart TD
    subgraph Inference
        model --> inference_loop
        DataLoader_Inference --> inference_loop
        inference_loop -->|iterate subjects| compute.step.step
    end
```