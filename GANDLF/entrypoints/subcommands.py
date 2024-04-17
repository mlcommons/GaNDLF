# import anonymizer command
from GANDLF.entrypoints.run import new_way as run_command

# import construct_csv command
# import collect_stats command
from GANDLF.entrypoints.patch_miner import new_way as patch_miner_command

from GANDLF.entrypoints.preprocess import new_way as preprocess_command

from GANDLF.entrypoints.verify_install import new_way as verify_install_command

from GANDLF.entrypoints.config_generator import new_way as config_generator_command

# import recover_config command
from GANDLF.entrypoints.deploy import new_way as deploy_command

from GANDLF.entrypoints.optimize_model import new_way as optimize_model_command

from GANDLF.entrypoints.generate_metrics import new_way as generate_metrics_command

# import debug_info command
# import split_csv command

cli_subcommands = {
    # TODO: add anonymizer command
    "run": run_command,
    # TODO: add construct-csv command
    # TODO: add collect-stats command
    "patch-miner": patch_miner_command,
    "preprocess": preprocess_command,
    "config-generator": config_generator_command,
    "verify-install": verify_install_command,
    # TODO: add recover-config command
    "deploy": deploy_command,
    "optimize-model": optimize_model_command,
    "generate-metrics": generate_metrics_command,
    # TODO: add debug-info command
    # TODO: add split-csv command
}
