<<<<<<< HEAD
## This example is to demonstrate the use of OpenVINO for GaNDLF pretrained classification models

### Requirements:
- Install GaNDLF following [Installation instructions](https://cbica.github.io/GaNDLF/setup)
- Install OpenVINO following [OpenVINO Installation](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- Install OpenVINO NNCF following [NNCF Installation](https://github.com/openvinotoolkit/nncf#installation)
- Active virtual environment and setup OpenVINO path

### Set up environment:
- Activate the conda virtual environment ```conda activate venv_gandlf```
- Setup OpenVINO path by running ```source /opt/intel/openvino_2021.4.689/bin/setupvars.sh```

### Usage: 
##### The scripts are located under: ```openvino_example/ClassificationSamples/scripts/```

#### Convert to OpenVINO FP32 models
- The current working directory is ```$ROOT_DIR```, and the pretrained 5-fold PyTorch classification models are located under ```$ROOT_DIR/DFU_experiments_vgg11_5fold_without_preprocess/```
- Use script ```run_convert_to_ov.sh``` to convert the pretrained 5-fold PyTorch classification models to OpenVINO model:
  ```
  bahs run_convert_to_ov.sh $ROOT_DIR
  ```
  After model conversion, we can find exported ONNX models under ```$ROOT_DIR/onnx``` and converted OpenVINO FP32 IR models under ```ROOT_DIR/ov_models```
 
#### POT quantization to INT8 models


=======
## This example is to demonstrate the use of OpenVINO for GaNDLF pretrained classification models

### Requirements:
- Install GaNDLF following [Installation instructions](https://cbica.github.io/GaNDLF/setup)
- Install OpenVINO following [OpenVINO Installation](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- Install OpenVINO NNCF following [NNCF Installation](https://github.com/openvinotoolkit/nncf#installation)
- Active virtual environment and setup OpenVINO path

### Set up environment:
- Activate the conda virtual environment ```conda activate venv_gandlf```
- Setup OpenVINO path by running ```source /opt/intel/openvino_2021.4.689/bin/setupvars.sh```

### Usage: 
##### The scripts are located under: ```openvino_example/ClassificationSamples/scripts/```
- The current working directory is ```$ROOT_DIR```, and the pretrained 5-fold PyTorch classification models are located under ```$ROOT_DIR/DFU_experiments_vgg11_5fold_without_preprocess/```
- Use script ```run_convert_to_ov.sh``` to convert the pretrained 5-fold PyTorch classification models to OpenVINO model:
  ```
  run_convert_to_ov.sh $ROOT_DIR
  ```
  After model conversion, we can find exported ONNX models under ```$ROOT_DIR/onnx``` and converted OpenVINO FP32 IR models under ```ROOT_DIR/ov_models```
>>>>>>> acd44e31950538f5d7f95d80234354ab1b2d6439
