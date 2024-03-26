import pickle

import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.deploy import new_way, old_way

from . import cli_runner, CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.deploy.run_deployment"
OLD_SCRIPT_NAME = "gandlf_deploy"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("model/"),
    TmpFile("model/parameters.pkl", content=pickle.dumps({"foo": "bar"})),
    TmpFile("model.file", content="123321"),
    TmpFile("config.yaml", content="baz: abc"),
    TmpDire("config_folder/"),
    TmpDire("mlcube_root/"),
    TmpFile("mlcube_root/mlcube.yaml"),
    TmpFile("tmp_test_entrypoint.py", content="print('Hello GaNDLF!')"),
    TmpFile("output.csv", content="foobar"),
    TmpNoEx("output_na.csv"),
    TmpDire("output/"),
    TmpNoEx("output_na/"),
    TmpNoEx("path_na"),
]
test_cases = [
    # =======================
    # Full command
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--model model/ --config config.yaml --target docker --mlcube-type model "
            + "--mlcube-root mlcube_root/ --output-dir output/ "
            + "--requires-gpu --entrypoint tmp_test_entrypoint.py",
            # tests short arg aliases
            "-m model/ -c config.yaml -t docker --mlcube-type model "
            + "-r mlcube_root/ -o output/ "
            + "-g -e tmp_test_entrypoint.py",
            # tests requires-gpu is True by default if not passed
            "-m model/ -c config.yaml -t docker --mlcube-type model "
            + "-r mlcube_root/ -o output/ "
            + "-e tmp_test_entrypoint.py",
        ],
        old_way_lines=[
            # full command
            "--model model/ --config config.yaml --target docker --mlcube-type model "
            + "--mlcube-root mlcube_root/ --outputdir output/ "
            + "--requires-gpu True --entrypoint tmp_test_entrypoint.py",
            # tests short arg aliases
            "-m model/ -c config.yaml -t docker --mlcube-type model "
            + "-r mlcube_root/ -o output/ "
            + "-g True -e tmp_test_entrypoint.py",
            # tests requires-gpu is True by default if not passed
            "-m model/ -c config.yaml -t docker --mlcube-type model "
            + "-r mlcube_root/ -o output/ "
            + "-e tmp_test_entrypoint.py",
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output/",
            "target": "docker",
            "mlcube_type": "model",
            "entrypoint_script": "tmp_test_entrypoint.py",
            "configfile": "config.yaml",
            "modeldir": "model/",
            "requires_gpu": True,
        },
    ),
    # =================
    # model-type checks
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # model_type is required and does not accept random values
            "-m model/ -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--model-type random_type -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/",
        ],
        old_way_lines=[
            "-m model/ -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--model-type random_type -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/",
        ],
    ),
    # ==================
    # Model MLCube
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # for model_type=model everything except entrypoint and config is required
            "--mlcube-type model -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--mlcube-type model -m model/ -c config.yaml -r mlcube_root/ -o output/",
            "--mlcube-type model -m model/ -c config.yaml -t docker -o output/",
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/",
            # also model should point to existing folder
            "--mlcube-type model -m model.file -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--mlcube-type model -m path_na -c config.yaml -t docker -r mlcube_root/ -o output/",
            # config if passed should point to file, not to folder
            "--mlcube-type model -m model/ -c path_na -t docker -r mlcube_root/ --o output/",
            "--mlcube-type model -m model/ -c config_folder/ -t docker -r mlcube_root/ -o output/",
            # the only supported target is docker, no random values
            "--mlcube-type model -m model/ -c config_folder/ -t stevedore -r mlcube_root/ -o output/",
            # model_root should point to existing folder
            "--mlcube-type model -m model/ -c config.yaml -t docker -r path_na -o output/",
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/mlcube.yaml -o output/",
            # output should point to a folder or to a non-existent path
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output.csv",
            # entrypoint, if passed, should point to existing file
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e path_na",
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e empty_folder/",
        ],
        old_way_lines=[
            # for model_type=model everything except config and entrypoint is required
            "--mlcube-type model -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--mlcube-type model -m model/ -c config.yaml -r mlcube_root/ -o output/",
            "--mlcube-type model -m model/ -c config.yaml -t docker -o output/",
            "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/",
            # also model should point to existing folder
            #          vvv---- in old way we do not check that model is dir (such a check happens later) ----vvv
            # "--mlcube-type model -m model.file -c config.yaml -t docker -r mlcube_root/ -o output/",
            #          vvv---- Also we do not check model path existence (such a check happens later) ----vvv
            # "--mlcube-type model -m path_na -c config.yaml -t docker -r mlcube_root/ -o output/",
            # config if passed should point to file, not to folder
            #          vvv---- in old way we don't check file existence ----vvv
            # "--mlcube-type model -m model/ -c path_na -t docker -r mlcube_root/ --o output/",
            #          vvv---- as well as that config is file ----vvv
            # "--mlcube-type model -m model/ -c config_folder/ -t docker -r mlcube_root/ -o output/",
            # the only supported target is docker, no random values
            #          vvv---- no such a check in old_way ----vvv
            # "--mlcube-type model -m model/ -c config_folder/ -t stevedore -r mlcube_root/ -o output/",
            # model_root should point to existing folder
            #          vvv---- no check for root existence in old_way (it happens later) ----vvv
            # "--mlcube-type model -m model/ -c config.yaml -t docker -r path_na -o output/",
            #          vvv---- no check root is dir in old_way (it happens later) ----vvv
            # "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/mlcube.yaml -o output/",
            # output should point to a folder or to non-existent path
            #          vvv---- despite this command fails, it fails when we try to create a file
            #          under output "folder" (while a real check that output is not a file happens later).
            #          Thus, as there is no real explicit check in old_way, this test is disabled
            #          ----vvv
            # "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output.csv",
            # entrypoint if passed should point to existing file
            #          vvv---- no check entrypoint exists in old_way (it happens later) ----vvv
            # "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e path_na",
            #          vvv---- no such a check in old_way ----vvv
            # "--mlcube-type model -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e empty_folder/",
        ],
    ),
    CliCase(  # Model + entrypoint
        should_succeed=True,
        new_way_lines=[
            # for model_type=model entrypoint is optional
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output/"
        ],
        old_way_lines=[
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output/"
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output/",
            "target": "docker",
            "mlcube_type": "model",
            "entrypoint_script": None,
            "configfile": "config.yaml",
            "modeldir": "model/",
            "requires_gpu": True,
        },
    ),
    CliCase(  # Model + config
        should_succeed=True,
        new_way_lines=[
            # for model_type=model config may be skipped; is restored from model then (`parameters.pkl`)
            "-m model/ -t docker --mlcube-type model -r mlcube_root/ -o output/"
        ],
        old_way_lines=[
            "-m model/ -t docker --mlcube-type model -r mlcube_root/ -o output/"
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output/",
            "target": "docker",
            "mlcube_type": "model",
            "entrypoint_script": None,
            "configfile": "output/original_config.yml",
            "modeldir": "model/",
            "requires_gpu": True,
        },
    ),
    # ================
    # Metrics MLCube
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # for model_type=metrics, model, config and entrypoint may be skipped
            "--mlcube-type metrics -t docker -r mlcube_root/ -o output/"
        ],
        old_way_lines=[
            # for model_type=metrics, model, config and entrypoint may be skipped
            "--mlcube-type metrics -t docker -r mlcube_root/ -o output/"
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output/",
            "target": "docker",
            "mlcube_type": "metrics",
            "entrypoint_script": None,
            "configfile": None,
            "modeldir": None,
            "requires_gpu": True,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # for model_type=metrics, target, mlcube_root and output are required
            "--mlcube-type metrics -m model/ -c config.yaml -r mlcube_root/ -o output/",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -o output/",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/",
            # model if passed should point to existing folder
            "--mlcube-type metrics -m model.file -c config.yaml -t docker -r mlcube_root/ -o output/",
            "--mlcube-type metrics -m path_na -c config.yaml -t docker -r mlcube_root/ -o output/",
            # config if passed should point to file, not to folder
            "--mlcube-type metrics -m model/ -c path_na -t docker -r mlcube_root/ --o output/",
            "--mlcube-type metrics -m model/ -c config_folder/ -t docker -r mlcube_root/ -o output/",
            # the only supported target is docker, no random values
            "--mlcube-type metrics -m model/ -c config_folder/ -t stevedore -r mlcube_root/ -o output/",
            # model_root should point to existing folder
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r path_na -o output/",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/mlcube.yaml -o output/",
            # output should point to a folder or to a non-existent path, not to a file
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ -o output.csv",
            # entrypoint, if passed, should point to existing file
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e path_na",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ --o output/ -e empty_folder/",
        ],
        old_way_lines=[
            # for model_type=metrics, target, mlcube_root and output are required
            "--mlcube-type metrics -m model/ -c config.yaml -r mlcube_root/ -o output/",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -o output/",
            "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/",
            # also model should point to existing folder
            #          vvv---- in old way we do not check if model is dir (such a check happens later) ----vvv
            # "--mlcube-type metrics -m model.file -c config.yaml -t docker -r mlcube_root/ -o output/",
            #          vvv---- Also we do not check model path existence (such a check happens later) ----vvv
            # "--mlcube-type metrics -m path_na -c config.yaml -t docker -r mlcube_root/ -o output/",
            # config if passed should point to file, not to folder
            #          vvv---- in old way we don't check file existence ----vvv
            # "--mlcube-type metrics -m model/ -c path_na -t docker -r mlcube_root/ --o output/",
            #          vvv---- as well as that config is file ----vvv
            # "--mlcube-type metrics -m model/ -c config_folder/ -t docker -r mlcube_root/ -o output/",
            # the only supported target is docker, no random values
            #          vvv---- no such a check in old_way ----vvv
            # "--mlcube-type metrics -m model/ -c config_folder/ -t stevedore -r mlcube_root/ -o output/",
            # model_root should point to existing folder
            #          vvv---- no check for root existence in old_way (it happens later) ----vvv
            # "--mlcube-type metrics -m model/ -c config.yaml -t docker -r path_na -o output/",
            #          vvv---- no check root is dir in old_way (it happens later) ----vvv
            # "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/mlcube.yaml -o output/",
            # output should point to a folder or to a non-existent path
            #          vvv---- despite this command fails, it fails when we try to create a file
            #          under output "folder" (while a real check that output is not a file happens later).
            #          Thus, as there is no real explicit check in old_way, this test is disabled
            #          ----vvv
            # "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ -o output.csv",
            # entrypoint, if passed, should point to existing file
            #          vvv---- no check entrypoint exists in old_way (it happens later) ----vvv
            # "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ -o output/ -e path_na",
            #          vvv---- no such a check in old_way ----vvv
            # "--mlcube-type metrics -m model/ -c config.yaml -t docker -r mlcube_root/ --o output/ -e empty_folder/",
        ],
    ),
    # ===============
    # Other options: requires_gpu
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # gpu may be disabled by passing --no-gpu
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output/ --no-gpu"
        ],
        old_way_lines=[
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output/ -g False"
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output/",
            "target": "docker",
            "mlcube_type": "model",
            "entrypoint_script": None,
            "configfile": "config.yaml",
            "modeldir": "model/",
            "requires_gpu": False,
        },
    ),
    CliCase(  # output folder may not exist (would be created)
        should_succeed=True,
        new_way_lines=[
            # gpu may be disabled by passing --no-gpu
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output_na/"
        ],
        old_way_lines=[
            "-m model/ -c config.yaml -t docker --mlcube-type model -r mlcube_root/ -o output_na/"
        ],
        expected_args={
            "mlcubedir": "mlcube_root/",
            "outputdir": "output_na/",
            "target": "docker",
            "mlcube_type": "model",
            "entrypoint_script": None,
            "configfile": "config.yaml",
            "modeldir": "model/",
            "requires_gpu": True,
        },
    ),
]


@pytest.mark.parametrize("case", test_cases)
def test_case(cli_runner: CliRunner, case: CliCase):
    run_test_case(
        cli_runner=cli_runner,
        file_system_config=test_file_system,
        case=case,
        real_code_function_path=MOCK_PATH,
        new_way=new_way,
        old_way=old_way,
        old_script_name=OLD_SCRIPT_NAME,
        patched_return_value=True,
    )
