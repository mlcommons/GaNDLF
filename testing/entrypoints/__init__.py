import importlib
from pathlib import Path
import pytest
from click import BaseCommand
from click.testing import CliRunner
from typing import Callable, Iterable, Any, Mapping, Optional, List, Union
import inspect
from dataclasses import dataclass
import sys
import shlex
import os
import shutil
from unittest.mock import patch


class ArgsExpander:
    def __init__(self, orig_func: Callable):
        self.orig_func = orig_func

    def normalize(self, args: Iterable[Any], kwargs: Mapping[str, Any]) -> dict:
        """
        Say, we have the following function:
        `def orig_func(param1: str, param2: str, train_flag=True)`
        mocked up our orig function with replica. After test is executed we see replica was called with some
        positional and some keyword args (say, it was `replica(foo, param2=bar)`).

        This function takes
            the list of positioned args `["foo"]`,
            dict of keyword args `{"param2": "bar"}`,
            signature of original function (that includes some params with default value also)
        and combine it together to receive the normalized dict of args that would be processed:
        {
            "param1": "foo",
            "param2": "bar",
            "train_flag": True
        }
        """
        # Get parameter names from the original function
        params = inspect.signature(self.orig_func).parameters
        arg_names = list(params.keys())

        # Build a dictionary of argument names to passed values
        # Start with positional arguments
        passed_args = {arg_names[i]: arg for i, arg in enumerate(args)}

        # Update the dictionary with keyword arguments
        passed_args.update(kwargs)

        # For any missing arguments that have defaults, add those to the dictionary
        for name, param in params.items():
            if name not in passed_args and param.default is not inspect.Parameter.empty:
                passed_args[name] = param.default

        return passed_args


# Fixture for Click's CliRunner to test Click commands
@pytest.fixture
def cli_runner():
    return CliRunner()


@dataclass
class CliCase:
    """
    Represent a specific case. All passed new way lines as well as old way lines should finally have exactly the same
    behavior and call a real logic function with `expected_args`.

    Args:
        should_succeed: if that console command should succeed or fail
        new_way_lines: list of str of the following format (in reality are passed as args to `gandlf` cli tool):
            '--input-dir input/ -c config.yaml -m rad --output-file output/'
            If skipped, new_way would not be tested.
        old_way_lines: list of str of the same format, but for old-fashioned format of cmd execution
            (say, via `gandlf_patchMiner`):
            '--inputDir input/ -c config.yaml -m rad --outputFile output/'
            If skipped, old_way would not be tested.
        expected_args: dict or params that should be finally passed to real logics code.
            Required, if `should_succeed`.

    """

    should_succeed: bool = True
    new_way_lines: list[str] = None
    old_way_lines: list[str] = None
    expected_args: dict = None


@dataclass
class _TmpPath:
    path: str


# it's not a typo in class name - I want to keep the same name len for dir / file / na
# for config to be more readable (paths are aligned in one column then)
@dataclass
class TmpDire(_TmpPath):
    pass


@dataclass
class TmpFile(_TmpPath):
    content: Optional[Union[str, bytes]] = None


@dataclass
class TmpNoEx(_TmpPath):
    pass


class TempFileSystem:
    """
    Given a dict of path -> path description (dir / file with content / na), creates
    the paths that are needed (dirs + files), and remove everything on the exit.
    For `na` files ensures they do not exist.

    If any of given paths already present on file system, then raises an error.

    By default, creates requested structure right in working directory.
    """

    def __init__(self, config: list[_TmpPath], root_dir=None):
        self.config = config
        self.root_dir = root_dir
        self.temp_paths: list[Path] = []

    def __enter__(self):
        try:
            self.setup_file_system()
        except Exception as e:
            self.cleanup()
            raise e
        return self

    def setup_file_system(self):
        for item in self.config:
            # no tmp files should exist beforehand as we will clean everything on exit
            path = Path(item.path)
            if self.root_dir:
                path = Path(self.root_dir) / path
            if path.exists():
                raise FileExistsError(
                    path,
                    "For temp file system all paths must absent beforehand as we remove everything "
                    "at the end.",
                )
            if isinstance(item, TmpDire):
                path.mkdir(parents=True, exist_ok=False)
            elif isinstance(item, TmpNoEx):
                pass  # we already ensured it not exists
            elif isinstance(item, TmpFile):
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(item.content, bytes):
                    with open(path, "wb") as fin:
                        fin.write(item.content)
                elif isinstance(item.content, str) or not item.content:
                    with open(path, "w") as fin:
                        if item.content:
                            fin.write(item.content)
                else:
                    raise ValueError(
                        f"Given tmp file has an invalid content (should be str or bytes): {item}"
                    )

            else:
                raise ValueError(f"Given tmp file entity is of invalid type: {item}")
            self.temp_paths.append(path)

    def cleanup(self):
        for path in reversed(self.temp_paths):
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                shutil.rmtree(path)
            elif not path.exists():
                pass
            else:
                raise ValueError(
                    f"wrong path {path}, not a dir, not a file. Cannot remove!"
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def args_diff(expected_args: dict[str, Any], actual_args: dict[str, Any]) -> list[str]:
    result = []
    for k in set(expected_args) | set(actual_args):
        if (
            k not in expected_args
            or k not in actual_args
            or expected_args[k] != actual_args[k]
        ):
            result.append(k)
    return result


def assert_called_properly(mock_func, expected_args: dict, args_normalizer):
    """Check that mock_func was called exactly once and passed args are identical to expected_args"""
    mock_func.assert_called_once()
    executed_call = mock_func.mock_calls[0]
    actual_args = args_normalizer.normalize(
        args=executed_call.args, kwargs=executed_call.kwargs
    )
    orig_args = expected_args
    expected_args = orig_args.copy()
    for arg, val in orig_args.items():
        # if expected arg is `...` , then we do not care about its actual value
        # just check the key presents in actual args
        if val is Ellipsis:
            assert arg in actual_args
            expected_args[arg] = actual_args[arg]

    assert expected_args == actual_args, (
        f"Function was not called with the expected arguments: {expected_args=} vs {actual_args=}, "
        f"diff {args_diff(expected_args, actual_args)}"
    )


def run_test_case(
    cli_runner: Optional[CliRunner],
    file_system_config: List[_TmpPath],
    case: CliCase,
    real_code_function_path: str,
    new_way: Optional[BaseCommand],
    old_way: Callable,
    old_script_name: str,
    patched_return_value: Any = None,
):
    module_path, func_name = real_code_function_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    real_code_function = getattr(module, func_name)

    args_normalizer = ArgsExpander(real_code_function)
    with patch(
        real_code_function_path, return_value=patched_return_value
    ) as mock_logic:
        # tests that all click commands trigger execution with expected args
        for new_line in case.new_way_lines or []:
            try:
                mock_logic.reset_mock()
                new_cmd = shlex.split(new_line)
                with TempFileSystem(file_system_config):
                    result = cli_runner.invoke(new_way, new_cmd)
                if case.should_succeed:
                    assert result.exit_code == 0
                    assert_called_properly(
                        mock_logic, case.expected_args, args_normalizer
                    )
                else:
                    assert result.exit_code != 0
            except BaseException:
                print(f"Test failed on the new case: {new_line}")
                print(f"Exception: {result.exception}")
                print(f"Exc info: {result.exc_info}")
                print(f"output: {result.output}")
                raise

        # tests that old way commands via `gandlf_*` script trigger the same expected_args
        for old_line in case.old_way_lines or []:
            try:
                mock_logic.reset_mock()
                argv = [old_script_name] + shlex.split(old_line)
                if case.should_succeed:
                    with patch.object(sys, "argv", argv), TempFileSystem(
                        file_system_config
                    ):
                        old_way()
                    assert_called_properly(
                        mock_logic, case.expected_args, args_normalizer
                    )
                else:
                    with (
                        # here we can list possible errors that are allowed to be raised.
                        # Any other raised exception would be treated as test fail
                        pytest.raises((SystemExit, AssertionError)) as e,
                        patch.object(sys, "argv", argv),
                        TempFileSystem(file_system_config),
                    ):
                        old_way()
                    if isinstance(e.value, SystemExit):
                        assert e.value.code != 0
            except BaseException:
                print(f"Test failed on the old case: {old_line}")
                raise
