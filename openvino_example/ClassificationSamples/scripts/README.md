## This example is to demonstrate the use of OpenVINO for GaNDLF pretrained classification models

### Requirements:
- Install GaNDLF following [Installation instructions](https://cbica.github.io/GaNDLF/setup)
- Install OpenVINO following [OpenVINO Installation](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- Install OpenVINO NNCF following [NNCF Installation](https://github.com/openvinotoolkit/nncf#installation)
- Active virtual environment and setup OpenVINO path

### Set up environment:
- Activate the conda virtual environment ```conda activate venv_gandlf```
- Setup OpenVINO path by running ```source /opt/intel/openvino_2021.4.689/bin/setupvars.sh -pyver 3.6```

### Usage: 
##### The scripts are located under: ```openvino_example/ClassificationSamples/scripts/```
- The current working directory is ```$ROOT_DIR```, and the pretrained 5-fold PyTorch classification models are located under ```$ROOT_DIR/DFU_experiments_vgg11_5fold_without_preprocess/```
- Use script ```run_convert_to_ov.sh``` to convert the pretrained 5-fold PyTorch classification models to OpenVINO model
- 
