#! /bin/sh

###############
#### Setup ####
###############

MODEL_MLCUBE_TEMPLATE=../../mlcube/model_mlcube/mlcube_medperf.yaml
METRICS_MLCUBE_TEMPLATE=../../mlcube/metrics_mlcube/mlcube_medperf.yaml

MODEL_MLCUBE_ENTRYPOINT=../../mlcube/model_mlcube/example_custom_entrypoint/getting_started_3d_rad_seg.py
METRICS_MLCUBE_ENTRYPOINT=../../mlcube/metrics_mlcube/example_custom_entrypoint/getting_started_3d_rad_seg.py

# Create a workspace
mkdir -p test_deploy
cd test_deploy

# Download the data
FILENAME=data.zip
wget https://drive.google.com/uc?id=1c4Yrv-jnK6Tk7Ne1HmMTChv-4nYk43NT --output-document=$FILENAME
unzip $FILENAME
mv data/3d_rad_segmentation .
rm $FILENAME
rm -rf data

# Setup the data CSV for training
echo "SubjectID,Channel_0,Label
001,3d_rad_segmentation/001/image.nii.gz,3d_rad_segmentation/001/mask.nii.gz
002,3d_rad_segmentation/002/image.nii.gz,3d_rad_segmentation/002/mask.nii.gz
003,3d_rad_segmentation/003/image.nii.gz,3d_rad_segmentation/003/mask.nii.gz
004,3d_rad_segmentation/004/image.nii.gz,3d_rad_segmentation/004/mask.nii.gz
005,3d_rad_segmentation/005/image.nii.gz,3d_rad_segmentation/005/mask.nii.gz
006,3d_rad_segmentation/006/image.nii.gz,3d_rad_segmentation/006/mask.nii.gz
007,3d_rad_segmentation/007/image.nii.gz,3d_rad_segmentation/007/mask.nii.gz
008,3d_rad_segmentation/008/image.nii.gz,3d_rad_segmentation/008/mask.nii.gz
009,3d_rad_segmentation/009/image.nii.gz,3d_rad_segmentation/009/mask.nii.gz
010,3d_rad_segmentation/010/image.nii.gz,3d_rad_segmentation/010/mask.nii.gz" >> data.csv

# Setup config file
cp ../../samples/config_getting_started_segmentation_rad3d.yaml .

##################
#### Training ####
##################

gandlf run \
  -c ./config_getting_started_segmentation_rad3d.yaml \
  -i ./data.csv \
  -m ./trained_model_output \
  -t \
  -d cpu

# remove data.csv to assume that we need a custom script with gandlf deploy
rm data.csv

################
#### deploy ####
################

echo "Starting model deploy..."
# deploy model
mkdir model_mlcube
cp $MODEL_MLCUBE_TEMPLATE model_mlcube/mlcube.yaml

gandlf deploy \
  -c ./config_getting_started_segmentation_rad3d.yaml \
  -m ./trained_model_output \
  --target docker \
  --mlcube-root ./model_mlcube \
  -o ./built_model_mlcube \
  --mlcube-type model \
  --no-gpu \
  --entrypoint $MODEL_MLCUBE_ENTRYPOINT

echo "Starting metrics deploy..."
# deploy metrics
mkdir metrics_mlcube
cp $METRICS_MLCUBE_TEMPLATE metrics_mlcube/mlcube.yaml

gandlf deploy \
  --target docker \
  --mlcube-root ./metrics_mlcube \
  -o ./built_metrics_mlcube \
  --mlcube-type metrics \
  --entrypoint $METRICS_MLCUBE_ENTRYPOINT

######################
#### run pipeline ####
######################

echo "Starting model pipeline run..."

mlcube run \
    --mlcube ./built_model_mlcube \
    --task infer \
    input-data=../../3d_rad_segmentation \
    output-path=../../predictions

echo "Starting metrics pipeline run..."

mlcube run \
    --mlcube ./built_metrics_mlcube \
    --task evaluate \
    predictions=../../predictions \
    labels=../../3d_rad_segmentation \
    output-file=../../results.yaml \
    config=../../config_getting_started_segmentation_rad3d.yaml


###############
#### check ####
###############

echo "Checking results..."

if [ -f "results.yaml" ]; then
    echo "Success"
    cd ..
    sudo rm -rf ./test_deploy/
else 
    echo "Failure"
    exit 1
fi
