## Related flowcharts 

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
        parameters([Parameters]) --> train_network[\compute.training_loop.train_network\]
        train_network -->|Create Compute Objects| create_pytorch_objects[\compute.generic.create_pytorch_objects\]
        Data_Training[(Data for Training)] --> step[Compute Step]
        model[Model] --> step
        optimizer[Optimizer] --> step
        step -->|Loss Backpropagation| optimizer
        create_pytorch_objects --> optimizer
        create_pytorch_objects --> scheduler[Scheduler]
        scheduler -->|Update Learning Rate| optimizer
        step -->|Latest Model| save_model[\utils.modelio.save_model\]
        step -->|Loss and Metrics| log_metrics[Training Logger]
        
        parameters <==>|Updated| create_pytorch_objects
        create_pytorch_objects --> Data_Training
        create_pytorch_objects --> model
    end
     
    subgraph Validation and Testing
        create_pytorch_objects --> Data_Validation[(Data for Validation)] -->|Validation Mode| validate_network[Validate Network]
        create_pytorch_objects --> Data_Testing[(Data for Testing)] -->|Testing Mode| validate_network
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
    subgraph Input
        Each_Sample[Each Sample] --> Input_GroundTruth[Input Ground Truth]
        Each_Sample --> Input_DataPoint[Input Data Point]
        Input_DataPoint -->|Forward Pass| model[Model]
    end
    
    subgraph Compute
        Input_GroundTruth -->|Compute Loss and Metric| type{Compute Loss and Metric}
        model --> Prediction[Prediction]
        Prediction -->|Compute Loss and Metric| type
        parameters([Parameters]) --> type
    end
    
    subgraph Output
        type -->|Training/Validation Mode| Loss_Return([Return Loss, Metrics, & Prediction])
        type -->|Testing Mode| Inference_Return([Return Only Prediction because no GT])
        Prediction --> Inference_Return
    end
```
