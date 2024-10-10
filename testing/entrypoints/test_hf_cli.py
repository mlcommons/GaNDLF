import os.path
import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.hf_hub_integration import new_way

from . import CliCase, run_test_case, TmpDire

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.hf_hub_integration.download_from_hub"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [TmpDire("./tmp_dir")]


test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--download --repo-id distilbert-base-uncased",
            # tests short arg aliases
            "-d -rid distilbert-base-uncased",
        ],
        expected_args={
            "repo_id": "distilbert-base-uncased",
            "revision": None,
            "cache_dir": None,
            "local_dir": None,
            "force_download": False,
            "token": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--download --repo-id distilbert-base-uncased --revision 6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411",
            # tests short arg aliases
            "-d -rid distilbert-base-uncased -rv 6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411",
        ],
        expected_args={
            "repo_id": "distilbert-base-uncased",
            "revision": "6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411",
            "cache_dir": None,
            "local_dir": None,
            "force_download": False,
            "token": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--download --repo-id distilbert-base-uncased --local-dir tmp_dir",
            # tests short arg aliases
            "-d -rid distilbert-base-uncased -ldir tmp_dir",
        ],
        expected_args={
            "repo_id": "distilbert-base-uncased",
            "revision": None,
            "cache_dir": None,
            "local_dir": os.path.normpath("tmp_dir"),
            "force_download": False,
            "token": None,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # full command
            "--repo-id distilbert-base-uncased ",
            # tests short arg aliases
            "-rid distilbert-base-uncased -ldir",
        ],
        expected_args={
            "repo_id": "distilbert-base-uncased",
            "revision": None,
            "cache_dir": None,
            "local_dir": None,
            "force_download": False,
            "token": None,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # full command
            "--download --repo-id distilbert-base-uncased --local-dir",
            # tests short arg aliases
            "-d -rid distilbert-base-uncased -ldir",
        ],
        expected_args={
            "repo_id": "distilbert-base-uncased",
            "revision": None,
            "cache_dir": None,
            "local_dir": os.path.normpath("tmp_dir"),
            "force_download": False,
            "token": None,
        },
    ),
]


@pytest.mark.parametrize("case", test_cases)
def test_case(cli_runner: CliRunner, case: CliCase):
    """This approach ensures that before passing file_system_config to run_test_case,
    you check its value and assign an appropriate default ([])."""
    file_system_config_ = test_file_system if test_file_system is not None else []

    run_test_case(
        cli_runner=cli_runner,
        file_system_config=file_system_config_,  # Default to empty list if no new_way_lines
        case=case,
        real_code_function_path=MOCK_PATH,
        new_way=new_way,  # Pass the real 'new_way' or set as needed
        old_way=None,
        old_script_name=None,
    )
