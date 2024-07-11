import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.generate_metrics import new_way, old_way
from . import CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.generate_metrics.generate_metrics_dict"
OLD_SCRIPT_NAME = "gandlf_generateMetrics"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("tmp_dir/"),
    TmpFile("input.csv", content="SubjectID,Target,Prediction\n1,1.0,1.5\n2,0.5,0.3"),
    TmpFile("config.yaml", content="foo: bar"),
    TmpFile("output.json"),
    TmpNoEx("output_na.csv"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--input-data input.csv --output-file output.json --config config.yaml",
            # tests short arg aliases
            "-i input.csv -o output.json -c config.yaml",
            # --raw-input param exists that do nothing
            "-i input.csv -o output.json -c config.yaml --raw-input 123321",
        ],
        old_way_lines=[
            "--inputdata input.csv --outputfile output.json --config config.yaml",
            "--data_path input.csv --output_path output.json --parameters_file config.yaml",
            "-i input.csv -o output.json -c config.yaml",
            # --raw-input param exists that do nothing
            "-i input.csv -o output.json -c config.yaml --rawinput 123321",
            "-i input.csv -o output.json -c config.yaml -rawinput 123321",
        ],
        expected_args={
            "input_csv": "input.csv",
            "config": "config.yaml",
            "outputfile": "output.json",
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # output is optional
            "-i input.csv -c config.yaml"
        ],
        old_way_lines=["-i input.csv -c config.yaml"],
        expected_args={
            "input_csv": "input.csv",
            "config": "config.yaml",
            "outputfile": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # output may not exist yet
            "-i input.csv -o output_na.json -c config.yaml"
        ],
        old_way_lines=["-i input.csv -o output_na.json -c config.yaml"],
        expected_args={
            "input_csv": "input.csv",
            "config": "config.yaml",
            "outputfile": "output_na.json",
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # input and config are required
            "-o output.json -c config.yaml",
            "-i input.csv -o output.json",
            # input, config should point to existing file, not dir
            "-i tmp_dir/ -o output.json -c config.yaml",
            "-i input.csv -o output.json -c path_na",
            "-i input.csv -o output.json -c tmp_dir/",
            # output if passed should not point to dir
            "-i input.csv -o tmp_dir/ -c config.yaml",
        ],
        old_way_lines=[
            # input and config are required
            "-o output.json -c config.yaml",
            "-i input.csv -o output.json",
            # input, config should point to existing file, not dir
            # "-i path_na -o output.json -c config.yaml",  # no check in old_way
            # "-i tmp_dir/ -o output.json -c config.yaml",   # no check in old_way
            # "-i input.csv -o output.json -c path_na",   # no check in old_way
            # "-i input.csv -o output.json -c tmp_dir/",   # no check in old_way
            # output if passed should not point to dir
            # "-i input.csv -o tmp_dir/ -c config.yaml",    # no check in old_way
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
