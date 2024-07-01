import pytest
import sys
from pathlib import Path
from .entrypoints import CliCase, run_test_case

parent = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent)

from update_version import main

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "update_version._update_version"
OLD_SCRIPT_NAME = "update_version.py"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = []
test_cases = [
    CliCase(
        should_succeed=True,
        old_way_lines=[
            # long and short versions
            "--old-version 0.18 --new-version 0.19",
            "-ov 0.18 -nv 0.19",
        ],
        expected_args={"old_version": "0.18", "new_version": "0.19"},
    ),
    CliCase(
        should_succeed=False,
        old_way_lines=[
            # both args are required
            "-ov 0.18",
            "-nv 0.19",
        ],
    ),
]


@pytest.mark.parametrize("case", test_cases)
def test_case(case: CliCase):
    run_test_case(
        cli_runner=None,
        file_system_config=test_file_system,
        case=case,
        real_code_function_path=MOCK_PATH,
        new_way=None,
        old_way=main,
        old_script_name=OLD_SCRIPT_NAME,
    )
