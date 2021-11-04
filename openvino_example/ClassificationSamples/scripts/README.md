## This example is to demonstrate the use of OpenVINO for GaNDLF pretrained classification models

### Requirements:
- Install GaNDLF following [Installation instructions](https://cbica.github.io/GaNDLF/setup)
- Install OpenVINO following [OpenVINO Installation](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- Install OpenVINO NNCF following [NNCF Installation](https://github.com/openvinotoolkit/nncf#installation)

### Set up environment:
- Activate the conda virtual environment ```conda activate venv_gandlf```
- Setup OpenVINO path by running ```source /opt/intel/openvino_2021.4.689/bin/setupvars.sh```

### Usage: 
##### The scripts are located under: ```openvino_example/ClassificationSamples/scripts/```

#### Convert to OpenVINO FP32 models
- The current model directory is ```$MODEL_DIR```, and the pretrained 5-fold PyTorch classification models are located under ```$MODEL_DIR$TORCH_MODEL_DIR```, 
  where $TORCH_MODEL_DIR is the relative directory to host the pretrained PyTorch models
- Use script ```run_convert_to_ov.sh``` to convert the pretrained 5-fold PyTorch classification models to OpenVINO model:
  ```
  bahs run_convert_to_ov.sh $MODEL_DIR $TORCH_MODEL_DIR
  ```
  After model conversion, we can find exported ONNX models under ```$MODEL_DIR/onnx``` and converted OpenVINO FP32 IR models under ```$MODEL_DIR/ov_models```
 
#### POT quantization to INT8 models
- The scripts for POT quantization is located under ```./quantization```
- Use script ```run_generate_data.sh``` to generate the patch data for both POT quantization and NNCF compression
```

```


