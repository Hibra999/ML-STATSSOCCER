from pathlib import Path

from src.agent.tools import ToolRuntime


def test_tool_runtime_delegates_to_cli_runner(tmp_path: Path):
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("ok")
        return 0

    tools = ToolRuntime(tmp_path, cli_runner=fake_cli)
    result = tools.run_cli(["model", "list", "epl-2018"])

    assert result.ok
    assert result.stdout.strip() == "ok"
    assert calls == [["model", "list", "epl-2018"]]


def test_tool_runtime_blocks_path_escape(tmp_path: Path):
    tools = ToolRuntime(tmp_path)

    result = tools.read_file("../secret.txt")

    assert result.exit_code == 1
    assert "escapes repository root" in result.stderr
