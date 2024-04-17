import shlex
import subprocess  # nosec B404
import pytest
from GANDLF.entrypoints.subcommands import cli_subcommands as gandlf_commands

old_way_entrypoints = [
    # old-way entrypoints
    "gandlf_anonymizer --help",
    "gandlf_collectStats --help",
    "gandlf_configGenerator --help",
    "gandlf_constructCSV --help",
    "gandlf_debugInfo --help",
    "gandlf_deploy --help",
    "gandlf_generateMetrics --help",
    "gandlf_optimizeModel --help",
    "gandlf_patchMiner --help",
    "gandlf_preprocess --help",
    "gandlf_recoverConfig --help",
    "gandlf_run --help",
    "gandlf_verifyInstall --help",
    "gandlf_splitCSV --help",
]

main_cli_command = ["gandlf --version"]
# new-way CLI subcommands
new_way_cli_commands = [f"gandlf {cmd} --help" for cmd in gandlf_commands.keys()]

# Combine static and dynamic commands
all_commands = old_way_entrypoints + main_cli_command + new_way_cli_commands


@pytest.mark.parametrize("command", all_commands)
def test_command_execution(command):
    print(f"Running '{command}'...")
    # Run the command and capture output, stderr, and exit status
    command_split = shlex.split(command)
    result = subprocess.run(  # nosec B603
        command_split,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Command '{command}' failed with output:\n{result.stdout}"
