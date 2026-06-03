from pathlib import Path

from src.agent.session import SessionStore


def test_session_store_persists_messages_commands_and_compaction(tmp_path: Path):
    store = SessionStore(tmp_path, session_id="test-session")
    store.add_message("user", "hello")
    store.add_message("assistant", "world")
    store.add_command("python cli.py league list", 0)
    store.add_active_skill("train")

    reloaded = SessionStore(tmp_path, session_id="test-session")

    assert reloaded.state.messages[0]["content"] == "hello"
    assert reloaded.state.commands[0]["exit_code"] == 0
    assert reloaded.state.active_skills == ["train"]

    reloaded.state.messages.extend({"role": "user", "content": f"m{i}", "timestamp": "now"} for i in range(20))
    summary = reloaded.compact(keep_last=2)

    assert "hello" in summary
    assert len(reloaded.state.messages) == 2
