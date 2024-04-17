import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.config_generator import new_way, old_way

from . import cli_runner, CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.config_generator.config_generator"
OLD_SCRIPT_NAME = "gandlf_configGenerator"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpFile("config.yaml", content="foo: bar"),
    TmpFile("strategy.yaml", content="baz: abc"),
    TmpFile("output.csv", content="col1,col2\n123,456\n"),
    TmpDire("output/"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--config config.yaml --strategy strategy.yaml --output output/",
            # tests short arg aliases
            "-c config.yaml -s strategy.yaml -o output/",
        ],
        old_way_lines=[
            "--config config.yaml --strategy strategy.yaml --output output/",
            "-c config.yaml -s strategy.yaml -o output/",
        ],
        expected_args={
            "base_config_path": "config.yaml",
            "strategy_path": "strategy.yaml",
            "output_dir": "output/",
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # config should exist
            "-c path_na -s strategy.yaml -o output/",
            # strategy should exist
            "-c config.yaml -s path_na -o output/",
            # config is required
            "-s strategy.yaml -o output/",
            # strategy is required
            "-c config.yaml -o output/",
            # output is required
            "-c config.yaml -s strategy.yaml",
            # output should be a dir, not file
            "-c config.yaml -s strategy.yaml -o output.csv",
        ],
        old_way_lines=[
            # "-c path_na -s strategy.yaml -o output/",  # in old way we do not check file existence
            # "-c config.yaml -s path_na -o output/",  # same
            "-s strategy.yaml -o output/",
            "-c config.yaml -o output/",
            "-c config.yaml -s strategy.yaml",
            # "-c config.yaml -s strategy.yaml -o output.csv",  # and do not check output is directory
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
