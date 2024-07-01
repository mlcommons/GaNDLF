from click.testing import CliRunner
from GANDLF.entrypoints.cli_tool import gandlf
from GANDLF.version import __version__


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(gandlf, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
