from click.testing import CliRunner
from GANDLF.entrypoints.cli_tool import gandlf


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(gandlf, ["--version"])
    assert result.exit_code == 0
