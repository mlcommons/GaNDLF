import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.update_version import new_way, old_way
from . import cli_runner, CliCase, run_test_case

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.update_version._update_version"
OLD_SCRIPT_NAME = "gandlf_updateVersion"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = []
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--old-version 0.18 --new-version 0.19",
            # short aliases
            "-ov 0.18 -nv 0.19",
        ],
        old_way_lines=[
            "--old_version 0.18 --new_version 0.19",
            "-ov 0.18 -nv 0.19",
        ],
        expected_args={"old_version": "0.18", "new_version": "0.19"},
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # both old-version and new-version are required
            "-ov 0.18",
            "-nv 0.19",
        ],
        old_way_lines=[
            "-ov 0.18",
            "-nv 0.19",
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
