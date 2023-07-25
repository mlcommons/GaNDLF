from .patch_extraction import patch_extraction
from .main_run import main_run
from .preprocess_and_save import preprocess_and_save
from .config_generator import config_generator
from .deploy import deploy_targets, mlcube_types, run_deployment
from .recover_config import recover_config
from .post_training_model_optimization import post_training_model_optimization
from .generate_metrics import generate_metrics_dict

from datetime import date

copyrightMessage = (
    "Contact: gandlf@mlcommons.org\n\n"
    + "This program is NOT FDA/CE approved and NOT intended for clinical use.\nCopyright (c) "
    + str(date.today().year)
    + " MLCommons. All rights reserved.\n\nCitation: https://arxiv.org/abs/2103.01006"
)
