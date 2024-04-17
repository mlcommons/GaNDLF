from GANDLF.entrypoints.anonymizer import new_way as anonymizer_command
from GANDLF.entrypoints.run import new_way as run_command
from GANDLF.entrypoints.construct_csv import new_way as construct_csv_command
from GANDLF.entrypoints.collect_stats import new_way as collect_stats_command
from GANDLF.entrypoints.patch_miner import new_way as patch_miner_command
from GANDLF.entrypoints.preprocess import new_way as preprocess_command
from GANDLF.entrypoints.verify_install import new_way as verify_install_command
from GANDLF.entrypoints.config_generator import new_way as config_generator_command
from GANDLF.entrypoints.recover_config import new_way as recover_config_command
from GANDLF.entrypoints.deploy import new_way as deploy_command
from GANDLF.entrypoints.optimize_model import new_way as optimize_model_command
from GANDLF.entrypoints.generate_metrics import new_way as generate_metrics_command
from GANDLF.entrypoints.debug_info import new_way as debug_info_command
from GANDLF.entrypoints.split_csv import new_way as split_csv_command


cli_subcommands = {
    "anonymizer": anonymizer_command,
    "run": run_command,
    "construct-csv": construct_csv_command,
    "collect-stats": collect_stats_command,
    "patch-miner": patch_miner_command,
    "preprocess": preprocess_command,
    "config-generator": config_generator_command,
    "verify-install": verify_install_command,
    "recover-config": recover_config_command,
    "deploy": deploy_command,
    "optimize-model": optimize_model_command,
    "generate-metrics": generate_metrics_command,
    "debug-info": debug_info_command,
    "split-csv": split_csv_command,
}
