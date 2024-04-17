import os.path
import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.split_csv import new_way, old_way

from . import cli_runner, CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.split_csv.split_data_and_save_csvs"
OLD_SCRIPT_NAME = "gandlf_splitCSV"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("input/"),
    TmpFile("input.csv", content="col1,col2\n123,456\n"),
    TmpFile("config.yaml", content="foo: bar"),
    TmpFile("config.txt", "@not-a-yaml-content"),
    TmpDire("config/"),
    TmpDire("output/"),
    TmpFile("output.csv", content="col1,col2\n123,456\n"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--input-csv input.csv --output-dir output/ --config config.yaml",
            # tests short arg aliases
            "-i input.csv -o output/ -c config.yaml",
        ],
        old_way_lines=[
            "--inputCSV input.csv --outputDir output/ --config config.yaml",
            "-i input.csv -o output/ -c config.yaml",
        ],
        expected_args={
            "input_data": os.path.normpath("input.csv"),
            "output_dir": os.path.normpath("output/"),
            "parameters": {"foo": "bar"},
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # tests that input, output, config are required
            "             -o output/ -c config.yaml",
            "-i input.csv            -c config.yaml",
            "-i input.csv -o output/               ",
            # tests that input points to existing file
            "-i input/ -o output/ -c config.yaml",
            "-i path_na -o output/ -c config.yaml",
            # tests that output points to existing dir
            "-i input.csv -o output.csv -c config.yaml",
            "-i input.csv -o path_na -c config.yaml",
            # tests that config points to existing yaml
            "-i input.csv -o output/ -c config.txt",
            "-i input.csv -o output/ -c config/",
            "-i input.csv -o output/ -c path_na",
        ],
        old_way_lines=[
            # tests that input, output, config are required
            "             -o output/ -c config.yaml",
            "-i input.csv            -c config.yaml",
            "-i input.csv -o output/               ",
            # tests that input points to existing file
            # "-i input/ -o output/ -c config.yaml",  # no check in old way
            # "-i path_na -o output/ -c config.yaml",  # no check in old way
            # tests that output points to existing dir
            # "-i input.csv -o output.csv -c config.yaml",  # no check in old way
            # "-i input.csv -o path_na -c config.yaml",  # no check in old way
            # tests that config points to existing yaml
            "-i input.csv -o output/ -c config.txt",
            "-i input.csv -o output/ -c config/",
            "-i input.csv -o output/ -c path_na",
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
