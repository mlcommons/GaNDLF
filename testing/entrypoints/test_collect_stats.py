import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.collect_stats import new_way, old_way

from . import CliCase, run_test_case, TmpNoEx, TmpDire, TmpFile

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.collect_stats._read_data_and_plot"
OLD_SCRIPT_NAME = "gandlf_collectStats"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_csv = "col1,col2\n1,100\n2,200"
test_file_system = [
    TmpDire("model_full/"),
    TmpFile("model_full/logs_training.csv", content=test_csv),
    TmpFile("model_full/logs_validation.csv", content=test_csv),
    TmpFile("model_full/logs_testing.csv", content=test_csv),
    TmpDire("model_no_test/"),
    TmpFile("model_no_test/logs_training.csv", content=test_csv),
    TmpFile("model_no_test/logs_validation.csv", content=test_csv),
    TmpDire("model_empty/"),
    TmpFile("file.txt", content="foobar"),
    TmpDire("output/"),
    TmpNoEx("output_na/"),
    TmpNoEx("path_na/"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--model-dir model_full/ --output-dir output/",
            # tests short arg aliases
            "-m model_full -o output/",
        ],
        old_way_lines=[
            "--modeldir model_full/ --outputdir output/",
            "-m model_full/ -o output/",
        ],
        expected_args={
            "training_logs_path": "model_full/logs_training.csv",
            "validation_logs_path": "model_full/logs_validation.csv",
            "testing_logs_path": "model_full/logs_testing.csv",
            "output_plot_path": "output/plot.png",
            "output_file": "output/data.csv",
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # test that it works without testing log
            "-m model_no_test -o output/"
        ],
        old_way_lines=["-m model_no_test/ -o output/"],
        expected_args={
            "training_logs_path": "model_no_test/logs_training.csv",
            "validation_logs_path": "model_no_test/logs_validation.csv",
            "testing_logs_path": None,
            "output_plot_path": "output/plot.png",
            "output_file": "output/data.csv",
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # test that output folder may not exist
            "-m model_full -o output_na/"
        ],
        old_way_lines=["-m model_full/ -o output_na/"],
        expected_args={
            "training_logs_path": "model_full/logs_training.csv",
            "validation_logs_path": "model_full/logs_validation.csv",
            "testing_logs_path": "model_full/logs_testing.csv",
            "output_plot_path": "output_na/plot.png",
            "output_file": "output_na/data.csv",
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # tests that input should exist
            "-m path_na -o output/",
            # tests that input is required
            "-o output/",
            # tests that output is required
            "-m model_full",
            # test that file is not accepted for input
            "-m file.txt -o output/",
            # test that file is not accepted for output
            "-m model_full -o file.txt",
        ],
        old_way_lines=[
            # "-m path_na -o output/",  # <- in old way model_dir is not required to exist
            # "-o output/", # <- ... or even be passed (code would fail immediately on data reading instead)
            # "-m model_full",  # <- same with output (if no output provided, code would fail on path operations)
            # "-m file.txt -o output/",  # <- same. No restrictions on model path
            # "-m model_full -o file.txt",  # <- same
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
