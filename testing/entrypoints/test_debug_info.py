import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.debug_info import new_way, old_way

from . import cli_runner, CliCase, run_test_case

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.debug_info._debug_info"
OLD_SCRIPT_NAME = "gandlf_debugInfo"


# subcommand is trivial, we just check both new_way and old_way run successfully
test_file_system = []
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[""],
        old_way_lines=[""],
        expected_args={}
    )
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
