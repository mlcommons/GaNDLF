import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.run import new_way, old_way
from . import CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.run.main_run"
OLD_SCRIPT_NAME = "gandlf_patchMiner"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
csv_content = "SubjectID,Target,Prediction\n1,1.0,1.5\n2,0.5,0.3"
test_file_system = [
    TmpFile("input.csv", content=csv_content),
    TmpFile("train.csv", content=csv_content),
    TmpFile("val.csv", content=csv_content),
    TmpDire("input/"),
    TmpFile("input/data.csv", content=csv_content),
    TmpFile("config.yaml", content="foo: bar"),
    TmpDire("config_dir/"),
    TmpDire("model/"),
    TmpFile("model.file"),
    TmpDire("output/"),
    TmpFile("output.csv"),
    TmpNoEx("output_na/"),
    TmpNoEx("path_na"),
]
# No tests for weird combinations: train + output-path, inference + reset/resume, as behavior is undefined
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command except --resume, --output-path
            "--config config.yaml --input-data input.csv --train --model-dir model/ "
            + "--device cuda --reset",
            # tests short arg aliases
            "-c config.yaml -i input.csv -t -m model/ -d cuda -rt",
            # test presence of --raw-input (and its uselessness)
            "-c config.yaml -i input.csv -t -m model/ -d cuda -rt --raw-input blabla",
        ],
        old_way_lines=[
            "--config config.yaml --inputdata input.csv --train True --modeldir model/ --device cuda --reset True",
            "--parameters_file config.yaml --data_path input.csv --train True --modeldir model/ --device cuda --reset True",
            "-c config.yaml -i input.csv -t True -m model/ -d cuda -rt True",
            # test presence of --raw-input (and its uselessness)
            "-c config.yaml -i input.csv -t True -m model/ -d cuda -rt True --rawinput blabla",
            "-c config.yaml -i input.csv -t True -m model/ -d cuda -rt True -rawinput blabla",
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cuda",
            "reset": True,
            "resume": False,
            "output_dir": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # --resume instead of --reset
            "-c config.yaml -i input.csv -t -m model/ -d cuda --resume",
            "-c config.yaml -i input.csv -t -m model/ -d cuda -rm",
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -m model/ -d cuda --resume True",
            "-c config.yaml -i input.csv -t True -m model/ -d cuda -rm True",
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cuda",
            "reset": False,
            "resume": True,
            "output_dir": None,
        },
    ),
    CliCase(  # inference mode + --output-path
        should_succeed=True,
        new_way_lines=[
            "-c config.yaml -i input.csv --infer -m model/ -d cuda --output-path output/",
            "-c config.yaml -i input.csv --infer -m model/ -d cuda -o output/",
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t False -m model/ -d cuda -o output/"
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": False,
            "device": "cuda",
            "reset": False,
            "resume": False,
            "output_dir": "output/",
        },
    ),
    CliCase(  # check that `model_dir` can be skipped (used output instead)
        should_succeed=True,
        new_way_lines=[
            "-c config.yaml -i input.csv --train -d cuda -o output/",
            "-c config.yaml -i input.csv --infer -d cuda -o output/",
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -d cuda -o output/",
            "-c config.yaml -i input.csv -t False -d cuda -o output/",
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "output/",
            "train_mode": ...,
            "device": "cuda",
            "reset": False,
            "resume": False,
            "output_dir": "output/",
        },
    ),
    CliCase(  # check that both output + model cannot be empty simultaneously
        should_succeed=False,
        new_way_lines=[
            "-c config.yaml -i input.csv --train -d cuda",
            "-c config.yaml -i input.csv --infer -d cuda",
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -d cuda",
            "-c config.yaml -i input.csv -t False -d cuda",
        ],
    ),
    CliCase(  # check device
        should_succeed=True,
        new_way_lines=[
            "-c config.yaml -i input.csv --train -m model/ -d cpu -o output/",
            "-c config.yaml -i input.csv --infer -m model/ -d cpu -o output/",
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -m model/ -d cpu -o output/",
            "-c config.yaml -i input.csv -t False -m model/ -d cpu -o output/",
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": ...,
            "device": "cpu",
            "reset": False,
            "resume": False,
            "output_dir": "output/",
        },
    ),
    CliCase(  # reset + resume simultaneously => disabling reset in favor of resume
        should_succeed=True,
        new_way_lines=[
            "-c config.yaml -i input.csv --train -m model/ -d cpu -o output/ -rt -rm"
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -m model/ -d cpu -o output/ -rt True -rm True"
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cpu",
            "reset": False,
            "resume": True,
            "output_dir": "output/",
        },
    ),
    CliCase(  # input data may point to folder with 'data.csv'
        should_succeed=True,
        new_way_lines=["-c config.yaml -i input/ --train -m model/ -d cpu"],
        old_way_lines=["-c config.yaml -i input/ -t True -m model/ -d cpu"],
        expected_args={
            "data_csv": "input/data.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cpu",
            "reset": False,
            "resume": False,
            "output_dir": None,
        },
    ),
    CliCase(  # input data may point to comma-separated list of csvs
        should_succeed=True,
        new_way_lines=["-c config.yaml -i train.csv,val.csv --train -m model/ -d cpu"],
        old_way_lines=["-c config.yaml -i train.csv,val.csv -t True -m model/ -d cpu"],
        expected_args={
            "data_csv": "train.csv,val.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cpu",
            "reset": False,
            "resume": False,
            "output_dir": None,
        },
    ),
    CliCase(  # output-path may point to non-existent path
        should_succeed=True,
        new_way_lines=[
            "-c config.yaml -i input.csv --train -m model/ -d cpu -o output_na/"
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -t True -m model/ -d cpu -o output_na/"
        ],
        expected_args={
            "data_csv": "input.csv",
            "config_file": "config.yaml",
            "model_dir": "model/",
            "train_mode": True,
            "device": "cpu",
            "reset": False,
            "resume": False,
            "output_dir": "output_na/",
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # config, input-data, train/infer, device are required
            "               -i input/ --train -m model/ -d cpu",
            "-c config.yaml           --train -m model/ -d cpu",
            "-c config.yaml -i input/         -m model/ -d cpu",
            "-c config.yaml -i input/ --train -m model/       ",
            # config should point to existing file
            "-c config_dir/ -i input/ --train -m model/ -d cpu",
            "-c path_na -i input/ --train -m model/ -d cpu",
            # output should not point to file
            "-c config.yaml -i input/ --train -d cpu -o output.csv",
            # model should not point to file
            "-c config.yaml -i input/ --train -m model.file -d cpu",
            # device should not support anything other beside cuda/cpu
            "-c config.yaml -i input/ --train -m model/ -d mps",
        ],
        old_way_lines=[
            # config, input-data, train/infer, device are required
            "               -i input/ -t True -m model/ -d cpu",
            "-c config.yaml           -t True -m model/ -d cpu",
            "-c config.yaml -i input/         -m model/ -d cpu",
            "-c config.yaml -i input/ -t True -m model/       ",
            # config should point to existing file
            "-c config_dir/ -i input/ -t True -m model/ -d cpu",
            "-c path_na -i input/ --train -m model/ -d cpu",
            # output should not point to file
            # "-c config.yaml -i input/ -t True -d cpu -o output.csv",  # no such check in old way
            # model should not point to file
            # "-c config.yaml -i input/ -t True -m model.file -d cpu",  # no such check in old way
            # device should not support anything other beside cuda/cpu
            # "-c config.yaml -i input/ -t True -m model/ -d mps",  # no such check in old way
        ],
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
    )
