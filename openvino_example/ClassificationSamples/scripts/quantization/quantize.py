import os
import numpy as np

from addict import Dict
from compression.graph import load_model, save_model
from compression.api import Metric
from compression.api.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.pipeline.initializer import create_pipeline

import argparse

parser = argparse.ArgumentParser(
    description="Quantizes an OpenVINO model to INT8.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--root_directory", default="/Share/Junwen/UPenn/DFU_Example_vgg11/ClassificationModel/ClassificationModel/", 
                    help="Root directory")
parser.add_argument("--model_directory", default="models/ov_models/",
                    help="Model directory")
parser.add_argument("--data_directory", default="scripts/quantization/data/vgg11/",
                    help="Data directory")
parser.add_argument("--maximum_metric_drop", default=1.0,
                    help="AccuracyAwareQuantization: Maximum allowed drop in metric")
parser.add_argument("--n_fold", default="0", type=str, 
                    help="n-fold")
parser.add_argument("--accuracy_aware_quantization",
                    help="use accuracy aware quantization",
                    action="store_true", default=False)

args = parser.parse_args()

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_data(input_data):
    npzfiles = np.load(input_data, allow_pickle = True )
    subsample_step = 200
    image, pred_label, gt_label = npzfiles.files
    images = npzfiles[image]
    length = images.shape[0]
    images = images[0:length:subsample_step, :, :, :, :]
    images = np.squeeze(images, axis=5)
    gt_label = npzfiles[gt_label]
    gt_label = gt_label[0:length:subsample_step]
    #gt_label = np.squeeze(gt_label)
    print(gt_label.shape)
    # gt_label = np.argmax(gt_label, axis=2)
    return(images, gt_label)

class DatasetsDataLoader(DataLoader):
 
    def __init__(self, config):
        super().__init__(config)
        self.images, self.gt_labels = read_data(str(config['data_source']))

    @property
    def size(self):
        return self.images.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        image = self.images[item,:,:,:,:]      
        label = self.gt_labels[item,:]
        return (item, label), image
    
class MyMetric(Metric):

    def __init__(self):
        super().__init__()
        self.name = "custom Metric - Accuracy"
        self._values = []
        self.round = 1

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        value = np.ravel(self._values).mean()
        print("Round #{}    Mean {} = {}".format(self.round, self.name, value))

        self.round += 1

        return {self.name: value}

    def update(self, outputs, labels):
        """ Updates prediction matches.

        Args:
            outputs: model output
            labels: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """

        pred = np.argmax(outputs)

        def accuracy(pred, truth):
            """
            Sorensen Dice score
            Measure of the overlap between the prediction and ground truth masks
            """
            if pred == truth:
                numerator = 1.0
            else:
                numerator = 0.0

            return numerator 

        metric = accuracy(labels, pred)
        self._values.append(metric)

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {self.name: {"direction": "higher-better", "type": ""}}
 
    
model_directory = os.path.join(args.root_directory, args.model_directory, args.n_fold)

# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': 'vgg11',
    'model': os.path.join(model_directory,  'vgg11_best.xml'),
    'weights': os.path.join(model_directory,  'vgg11_best.bin')
    })

print(model_config)

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

dataset_config = Dict({
    'data_source': os.path.join(args.root_directory, args.data_directory, args.n_fold, "train/patch_samples.npz")# Path to input data for quantization
})

print(dataset_config)

# Quantization algorithm settings

default_quantization_algorithm = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            #"stat_subset_size": 10
        }
    }
]


accuracy_aware_quantization_algorithm = [
    {
        "name": "AccuracyAwareQuantization", # compression algorithm name
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            "stat_subset_size": 10,
            "metric_subset_ratio": 0.5, # A part of the validation set that is used to compare full-precision and quantized models
            "ranking_subset_size": 300, # A size of a subset which is used to rank layers by their contribution to the accuracy drop
            "max_iter_num": 10,    # Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
            "maximal_drop": float(args.maximum_metric_drop),      # Maximum metric drop which has to be achieved after the quantization
            "drop_type": "absolute",    # Drop type of the accuracy metric: relative or absolute (default)
            "use_prev_if_drop_increase": True,     # Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
            "base_algorithm": "DefaultQuantization" # Base algorithm that is used to quantize model at the beginning
        }
    }
]


# Load the model.
model = load_model(model_config)
metric = MyMetric()

# Initialize the data loader.
data_loader = DatasetsDataLoader(dataset_config)

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, metric)

# Create a pipeline of compression algorithms.
if args.accuracy_aware_quantization:
    # https://docs.openvinotoolkit.org/latest/_compression_algorithms_quantization_accuracy_aware_README.html
    print(bcolors.BOLD + "Accuracy-aware quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(accuracy_aware_quantization_algorithm, engine)
else:
    print(bcolors.BOLD + "Default quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(default_quantization_algorithm, engine)

# print(pipeline)
metric_results_FP32 = pipeline.evaluate(model)
print("TEST: ",  metric_results_FP32)
# Execute the pipeline.

compressed_model = pipeline.run(model)


# Save the compressed model.
int8_directory = os.path.join(model_directory, 'INT8')
save_model(compressed_model, int8_directory)

metric_results_INT8 = pipeline.evaluate(compressed_model)

# print metric value
if metric_results_FP32:
    for name, value in metric_results_FP32.items():
        print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)

if metric_results_INT8:
    for name, value in metric_results_INT8.items():
        print(bcolors.OKBLUE + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)


print(bcolors.BOLD + "\nThe INT8 version of the model has been saved to the directory ".format(int8_directory) + \
    bcolors.HEADER + "{}\n".format(int8_directory) + bcolors.ENDC)
