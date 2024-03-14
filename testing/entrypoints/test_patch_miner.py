import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.patch_miner import new_way, old_way
from . import cli_runner, CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.patch_miner.patch_extraction"
OLD_SCRIPT_NAME = "gandlf_patchMiner"

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
            "--input-csv input.csv --output-path output/ --config config.yaml",
            # tests short arg aliases
            "-i input.csv -o output/ -c config.yaml",
        ],
        old_way_lines=[
            "--input_CSV input.csv --output_path output/ --config config.yaml",
            "-i input.csv -o output/ -c config.yaml",
        ],
        expected_args={
            "input_path": "input.csv",
            "config": "config.yaml",
            "output_path": "output/",
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # config is optional
            "-i input.csv -o output/",
        ],
        old_way_lines=[
            "-i input.csv -o output/",
        ],
        expected_args={
            "input_path": "input.csv",
            "config": None,
            "output_path": "output/",
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # output may not exist yet
            "-i input.csv -o output_na/",
        ],
        old_way_lines=[
            "-i input.csv -o output_na/",
        ],
        expected_args={
            "input_path": "input.csv",
            "config": None,
            "output_path": "output_na/",
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # input and output are required
            "-o output/",
            "-i input.csv",
            # input should point to existing file, not dir
            "-i path_na -o output/ -c config.yaml",
            "-i tmp_dir/ -o output/ -c config.yaml",
            # config if passed should point to existing file, not dir
            "-i input.csv -o output/ -c path_na",
            "-i input.csv -o output/ -c tmp_dir/",
            # output should point to dir, not file
            "-i input.csv -o output.csv -c config.yaml",
        ],
        old_way_lines=[
            # input and output are required
            "-o output/",
            "-i input.csv",
            # input should point to existing file, not dir
            # "-i path_na -o output/ -c config.yaml",  # no check in old_way
            # "-i tmp_dir/ -o output/ -c config.yaml",  # no check in old_way
            # config if passed should point to existing file, not dir
            # "-i input.csv -o output/ -c path_na",  # no check in old_way
            # "-i input.csv -o output/ -c tmp_dir/",  # no check in old_way
            # output should point to dir, not file
            # "-i input.csv -o output.csv -c config.yaml",  # no check in old_way
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
