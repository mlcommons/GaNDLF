import os

import pytest
from click.testing import CliRunner

from GANDLF.entrypoints.construct_csv import new_way, old_way

from . import cli_runner, CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "GANDLF.entrypoints.construct_csv.writeTrainingCSV"
OLD_SCRIPT_NAME = "gandlf_constructCSV"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("input/"),
    TmpFile("channels_str.yaml", content="channels: _yaml1.gz,_yaml2.gz"),
    TmpFile("channels_list.yaml", content="channels:\n  - _yaml1.gz\n  - _yaml2.gz"),
    TmpFile(
        "channels_labels.yaml", content="channels: _yaml1.gz,_yaml2.gz\nlabel: _yaml.gz"
    ),
    TmpFile("output.csv", content="foobar"),
    TmpNoEx("output_na.csv"),
    TmpDire("output/"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # full command
            "--input-dir input/ --channels-id _t1.nii.gz,_t2.nii.gz --label-id _seg.nii.gz --output-file output.csv --relativize-paths",
            # tests short arg aliases
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -l _seg.nii.gz -o output.csv -r",
        ],
        old_way_lines=[
            "--inputDir input/ --channelsID _t1.nii.gz,_t2.nii.gz --labelID _seg.nii.gz --outputFile output.csv --relativizePaths True",
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -l _seg.nii.gz -o output.csv -r True",
        ],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_t1.nii.gz,_t2.nii.gz",
            "labelID": "_seg.nii.gz",
            "outputFile": os.path.normpath("output.csv"),
            "relativizePathsToOutput": True,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # -r by default False
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -l _seg.nii.gz -o output.csv"
        ],
        old_way_lines=[
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -l _seg.nii.gz -o output.csv -r False",
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -l _seg.nii.gz -o output.csv",
        ],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_t1.nii.gz,_t2.nii.gz",
            "labelID": "_seg.nii.gz",
            "outputFile": os.path.normpath("output.csv"),
            "relativizePathsToOutput": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # channels may be read from yaml (str or list)
            "-i input/ -c _yaml1.gz,_yaml2.gz -l _seg.nii.gz -o output.csv",
            "-i input/ -c channels_str.yaml -l _seg.nii.gz -o output.csv",
            "-i input/ -c channels_list.yaml -l _seg.nii.gz -o output.csv",
        ],
        old_way_lines=[
            "-i input/ -c _yaml1.gz,_yaml2.gz -l _seg.nii.gz -o output.csv",
            "-i input/ -c channels_str.yaml -l _seg.nii.gz -o output.csv",
            "-i input/ -c channels_list.yaml -l _seg.nii.gz -o output.csv",
        ],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_yaml1.gz,_yaml2.gz",
            "labelID": "_seg.nii.gz",
            "outputFile": os.path.normpath("output.csv"),
            "relativizePathsToOutput": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # label-id can be defined in channels yaml also; arg value is skipped then
            "-i input/ -c channels_labels.yaml -l _arg_no_use.gz -o output.csv",
            "-i input/ -c channels_labels.yaml -o output.csv",
        ],
        old_way_lines=[
            "-i input/ -c channels_labels.yaml -l _arg_no_use.gz -o output.csv",
            "-i input/ -c channels_labels.yaml -o output.csv",
        ],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_yaml1.gz,_yaml2.gz",
            "labelID": "_yaml.gz",
            "outputFile": os.path.normpath("output.csv"),
            "relativizePathsToOutput": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # label-id can be skipped totally
            "-i input/ -c _yaml1.gz,_yaml2.gz -o output.csv",
            "-i input/ -c channels_str.yaml -o output.csv",
            "-i input/ -c channels_list.yaml -o output.csv",
        ],
        old_way_lines=[
            "-i input/ -c _yaml1.gz,_yaml2.gz -o output.csv",
            "-i input/ -c channels_str.yaml -o output.csv",
            "-i input/ -c channels_list.yaml -o output.csv",
        ],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_yaml1.gz,_yaml2.gz",
            "labelID": None,
            "outputFile": os.path.normpath("output.csv"),
            "relativizePathsToOutput": False,
        },
    ),
    CliCase(
        should_succeed=True,
        new_way_lines=[
            # output may not exist
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -o output_na.csv"
        ],
        old_way_lines=["-i input/ -c _t1.nii.gz,_t2.nii.gz -o output_na.csv"],
        expected_args={
            "inputDir": os.path.normpath("input/"),
            "channelsID": "_t1.nii.gz,_t2.nii.gz",
            "labelID": None,
            "outputFile": os.path.normpath("output_na.csv"),
            "relativizePathsToOutput": False,
        },
    ),
    CliCase(
        should_succeed=False,
        new_way_lines=[
            # input should be passed & exist
            "-i path_na -c _t1.nii.gz,_t2.nii.gz -o output.csv",
            "-c channels_str.yaml -o output.csv",
            # channel should be passed; file may not exist, value is treated as list of suffixes
            "-i input/ -o output.csv",
            # output should be passed and should not point to existing dir (file only is supported)
            "-i input/ -c _t1.nii.gz,_t2.nii.gz",
            "-i input/ -c _t1.nii.gz,_t2.nii.gz -o output/",
        ],
        old_way_lines=[
            # input should be passed & exist
            # "-i path_na -c _t1.nii.gz,_t2.nii.gz -o output.csv", # no checks for existence in old way
            "-c channels_str.yaml -o output.csv",
            # channel should be passed
            "-i input/ -o output.csv",
            # output should be passed
            "-i input/ -c _t1.nii.gz,_t2.nii.gz",
            # "-i input/ -c _t1.nii.gz,_t2.nii.gz -o output/",  # no checks for file/dir in old way
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
