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
FILENAME=y8162xkq1zz5555ye3pwadry2m2e39bs.zip
wget https://upenn.box.com/shared/static/$FILENAME
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

gandlf_run \
  -c ./config_getting_started_segmentation_rad3d.yaml \
  -i ./data.csv \
  -m ./trained_model_output \
  -t True \
  -d cpu

# remove data.csv to assume that we need a custom script with gandlf deploy
rm data.csv

################
#### deploy ####
################

# deploy model
mkdir model_mlcube
cp $MODEL_MLCUBE_TEMPLATE model_mlcube/mlcube.yaml

gandlf_deploy \
  -c ./config_getting_started_segmentation_rad3d.yaml \
  -m ./trained_model_output \
  --target docker \
  --mlcube-root ./model_mlcube \
  -o ./built_model_mlcube \
  --mlcube-type model \
  -g False \
  --entrypoint $MODEL_MLCUBE_ENTRYPOINT

# deploy metrics
mkdir metrics_mlcube
cp $METRICS_MLCUBE_TEMPLATE metrics_mlcube/mlcube.yaml

gandlf_deploy \
  --target docker \
  --mlcube-root ./metrics_mlcube \
  -o ./built_metrics_mlcube \
  --mlcube-type metrics \
  --entrypoint $METRICS_MLCUBE_ENTRYPOINT

######################
#### run pipeline ####
######################

mlcube run \
    --mlcube ./built_model_mlcube \
    --task infer \
    data_path=../../3d_rad_segmentation \
    output_path=../../predictions

mlcube run \
    --mlcube ./built_metrics_mlcube \
    --task evaluate \
    predictions=../../predictions \
    labels=../../3d_rad_segmentation \
    output_path=../../results.yaml \
    parameters_file=../../config_getting_started_segmentation_rad3d.yaml


###############
#### check ####
###############

if [ -f "results.yaml" ]; then
    echo "Success"
    cd ..
    sudo rm -rf ./test_deploy/
else 
    echo "Failure"
    exit 1
fi
