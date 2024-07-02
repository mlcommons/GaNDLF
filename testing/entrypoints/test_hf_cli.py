from click.testing import CliRunner
from GANDLF.entrypoints.cli_tool import gandlf

DUMMY_MODEL_ID = "julien-c/dummy-unknown"


class TestDownloadCommand:
    def test_download_basic(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "download", DUMMY_MODEL_ID])
        assert result.exit_code == 0


class TestEnvCommand:
    def test_env(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "env"])
        assert result.exit_code == 0


class TestCacheCommand:
    def test_scan_cache_basic(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "scan-cache"])
        assert result.exit_code == 0

    def test_scan_cache_verbose(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "scan-cache", "-v"])
        assert result.exit_code == 0

    def test_scan_cache_with_dir(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "scan-cache", "--dir", "something"])
        assert result.exit_code == 0

    def test_scan_cache_ultra_verbose(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "scan-cache", "-vvv"])
        assert result.exit_code == 0


class TestTagCommands:
    def test_tag_list_basic(self) -> None:
        runner = CliRunner()
        result = runner.invoke(gandlf, ["hf", "tag", "--list", DUMMY_MODEL_ID])
        assert result.exit_code == 0
