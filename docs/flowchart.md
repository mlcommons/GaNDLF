## Flowchart of GaNDLF

```mermaid
flowchart TD
    subgraph Input [Start]
        InputData -->|pandas csv| DataFrame
        InputConfig --> ConfigManager --> parameters
    end

    subgraph Data Processing
        DataFrame --> |Training| TrainingManager --> Data_Training --> training_loop
        DataFrame --> |Inference| InferenceManager --> Data_Inference --> inference_loop
        parameters --> TrainingManager
        parameters --> InferenceManager
    end

    subgraph Training Loop
        training_loop --> create_pytorch_objects
        parameters --> create_pytorch_objects
        create_pytorch_objects --> model
        create_pytorch_objects --> optimizer
        create_pytorch_objects --> scheduler
        create_pytorch_objects --> DataLoader_Training
        create_pytorch_objects --> DataLoader_Validation
        create_pytorch_objects --> weights
        create_pytorch_objects -->|updated| parameters
        model -->|initialized model| utils.modelio.save_mode
    end

    subgraph Training
        model --> train_network
        DataLoader_Training --> train_network
        optimizer --> train_network
        parameters --> train_network
        train_network -->|iterate subjects| compute.step.step
        compute.step.step -->|latest model| utils.modelio.save_mode
    end

    subgraph Validation
        model --> validate_network
        DataLoader_Validation --> validate_network
        optimizer --> validate_network
        validate_network -->|if validation loss improves| utils.modelio.save_mode
    end

```