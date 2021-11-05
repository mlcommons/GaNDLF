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
  bash run_convert_to_ov.sh $MODEL_DIR $TORCH_MODEL_DIR
  ```
  After model conversion, we can find exported ONNX models under ```$MODEL_DIR/onnx``` and converted OpenVINO FP32 IR models under ```$MODEL_DIR/ov_models```
 
#### POT quantization to INT8 models
- The scripts for POT quantization is located under ```./quantization```
- Use script ```run_generate_data.sh``` to generate the patch data for both POT quantization and NNCF compression
The script ```run_generate_data.sh``` takes 3 required positional arguments and 1 optional positional argument:
  - The first required positional argument is the working directory $ROOT_DIR. It includes sub-directories to hold the models and the data
  - The second required positional argument is the relative PyTorch model directory $PYTORCH_MODEL, so the model is reside under $ROOT_DIR$PYTORCH_MODEL
  - The third required positional argument is the relative data directory $DATA, so the data is reside under $ROOT_DIR$DATA. Under data directory $ROOT_DIR$DATA, we have the following sub-directorys
      - csv_files: it holds the data_training.csv and data_validation_csv according to GaNDLF's csv file format
      - patch_data: the generated numpy data files that include patch data and the corresponding labels
  - The fourth argument is optional. It specifies the data sampling rate. When this argument is specified, the user can get a sub-sample of the training data using the given sub-sampling rate. 
Here is how we run the patch data generation script:
```
bash run_generate_data.sh $ROOT_DIR $PYTORCH_MODEL $DATA [$SAMPLEING_RATE]
```
- Once we generate the patch data, the user can run the ```quantize.py``` to do the POT quantization to generate the quantized INT8 model. 
We will need to specify whether we want to accuracy_aware_quantization as well as which model we are quantizing. 
``` 
python3.6 quantize.py --model_directory ../../../../../ClassificationModel/ClassificationModel/models/ov_models/ --data_directory ../../../../../ClassificationModel/ClassificationModel/data/patch_data/ --accuracy_aware_quantization
```
Input to the quantize.py include the follows:
	- model_directory: the .xml file for the OpenVINO FP32 model
        - data_directory: the data file that holds the patches of the training data
        - accuracy_aware_quantization: whether we want to use accuracy_aware_quantization
        - subsample_step: the sampling rate to the patch data, default is 200, which means 1/200 samples from the training data are used for quantization




