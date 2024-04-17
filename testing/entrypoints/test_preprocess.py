import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.preprocess import new_way, old_way
from . import CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.preprocess.preprocess_and_save"
OLD_SCRIPT_NAME = "gandlf_preprocess"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("tmp_dir/"),
    TmpFile("input.csv", content="SubjectID,Target,Prediction\n1,1.0,1.5\n2,0.5,0.3"),
    TmpFile("config.yaml", content="foo: bar"),
    TmpDire("output/"),
    TmpFile("output.csv"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--config config.yaml --input-data input.csv --output output/ --label-pad constant --apply-augs --crop-zero",
            # tests short arg aliases
            "-c config.yaml -i input.csv -o output/ -l constant -a -z",
            # checks --label-pad is optional with `constant` default value
            "-c config.yaml -i input.csv -o output/ -a -z",
        ],
        old_way_lines=[
            "--config config.yaml --inputdata input.csv --output output/ --labelPad constant --applyaugs True --cropzero True",
            "-c config.yaml -i input.csv -o output/ -l constant -a True -z True",
            "-c config.yaml -i input.csv -o output/ -a True -z True",
        ],
        expected_args={
            "config_file": "config.yaml",
            "data_csv": "input.csv",
            "output_dir": "output/",
            "label_pad_mode": "constant",
            "applyaugs": True,
            "apply_zero_crop": True,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # tests flags (--apply-augs, --crop-zero)
            "-c config.yaml -i input.csv -o output/"
        ],
        old_way_lines=[
            "-c config.yaml -i input.csv -o output/",
            # vvv--- don't work as any passed value is transformed to `True`
            # "-c config.yaml -i input.csv -o output/ -a False -z False",
            # "-c config.yaml -i input.csv -o output/ -a False -z False",
        ],
        expected_args={
            "config_file": "config.yaml",
            "data_csv": "input.csv",
            "output_dir": "output/",
            "label_pad_mode": "constant",
            "applyaugs": False,
            "apply_zero_crop": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # tests --label-pad
            "-c config.yaml -i input.csv -o output/ -l mean"
        ],
        old_way_lines=["-c config.yaml -i input.csv -o output/ -l mean"],
        expected_args={
            "config_file": "config.yaml",
            "data_csv": "input.csv",
            "output_dir": "output/",
            "label_pad_mode": "mean",
            "applyaugs": False,
            "apply_zero_crop": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # output may not exist yet
            "-i input.csv -o output_na/ -c config.yaml"
        ],
        old_way_lines=["-i input.csv -o output_na/ -c config.yaml"],
        expected_args={
            "config_file": "config.yaml",
            "data_csv": "input.csv",
            "output_dir": "output_na/",
            "label_pad_mode": "constant",
            "applyaugs": False,
            "apply_zero_crop": False,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # input, output and config are required
            "-o output/ -c config.yaml",
            "-i input.csv -c config.yaml",
            "-i input.csv -o output/",
            # input should point to existing file, not dir
            "-i path_na -o output/ -c config.yaml",
            "-i tmp_dir/ -o output/ -c config.yaml",
            # config should point to existing file, not dir
            "-i input.csv -o output/ -c path_na",
            "-i input.csv -o output/ -c tmp_dir/",
            # output should point to dir, not file
            "-i input.csv -o output.csv -c config.yaml",
        ],
        old_way_lines=[
            # input, output and config are required
            "-o output/ -c config.yaml",
            "-i input.csv -c config.yaml",
            "-i input.csv -o output/",
            # input should point to existing file, not dir
            # "-i path_na -o output/ -c config.yaml",  # no check in old way
            # "-i tmp_dir/ -o output/ -c config.yaml",  # no check in old way
            # config should point to existing file, not dir
            # "-i input.csv -o output/ -c path_na",  # no check in old way
            # "-i input.csv -o output/ -c tmp_dir/",  # no check in old way
            # output should point to dir, not file
            # "-i input.csv -o output.csv -c config.yaml",  # no check in old way
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
