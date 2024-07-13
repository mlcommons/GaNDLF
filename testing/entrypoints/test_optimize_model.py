import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.optimize_model import new_way, old_way
from . import CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.optimize_model.post_training_model_optimization"
OLD_SCRIPT_NAME = "gandlf_optimizeModel"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("tmp_dir/"),
    TmpFile("model.pth.tar", content="123321"),
    TmpFile("config.yaml", content="foo: bar"),
    TmpNoEx("path_na"),
    TmpDire("output/"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command with output
            "--model model.pth.tar --config config.yaml --output-path output/",
            # tests short arg aliases
            "-m model.pth.tar -c config.yaml -o output/",
        ],
        old_way_lines=[
            "--model model.pth.tar --config config.yaml --output_path output/",
            "-m model.pth.tar -c config.yaml -o output/",
        ],
        expected_args={
            "model_path": "model.pth.tar",
            "config_path": "config.yaml",
            "output_path": "output/",
            "output_dir": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--model model.pth.tar --config config.yaml",
            # tests short arg aliases
            "-m model.pth.tar -c config.yaml",
        ],
        old_way_lines=[
            "--model model.pth.tar --config config.yaml",
            "-m model.pth.tar -c config.yaml",
        ],
        expected_args={
            "model_path": "model.pth.tar",
            "config_path": "config.yaml",
            "output_dir": None,
            "output_path": None,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # config is optional
            "-m model.pth.tar"
        ],
        old_way_lines=["-m model.pth.tar"],
        expected_args={
            "model_path": "model.pth.tar",
            "config_path": None,
            "output_path": None,
            "output_dir": None,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # model is required
            "-c config.yaml",
            # input, config should point to existing file, not dir
            "-m path_na -c config.yaml",
            "-m tmp_dir/ -c config.yaml",
            "-m model.pth.tar -c path_na",
            "-m model.pth.tar -c tmp_dir/",
        ],
        old_way_lines=[
            # model is required
            "-c config.yaml",
            # input, config should point to existing file, not dir
            # "-m path_na -c config.yaml",  # no check in old way
            # "-m tmp_dir/ -c config.yaml",  # no check in old way
            # "-m model.pth.tar -c path_na",  # no check in old way
            # "-m model.pth.tar -c tmp_dir/",  # no check in old way
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
