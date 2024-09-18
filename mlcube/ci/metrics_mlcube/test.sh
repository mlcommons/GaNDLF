# classification
rm ./mlcube/workspace/results.yaml
mlcube run --mlcube ./mlcube \
    --task evaluate \
    predictions=../../test_classification/predictions \
    labels=../../test_classification/labels \
    config=../../test_classification/config.yaml

# # segmentation (FAILS BECAUSE OF RELATIVE PATHS)
# rm ./mlcube/workspace/results.yaml
# mlcube run --mlcube ./mlcube \
#     --task evaluate \
#     predictions=../../test_segmentation/predictions \
#     labels=../../test_segmentation/labels \
#     config=../../test_segmentation/config.yaml
