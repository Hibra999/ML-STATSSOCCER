from src.agent.permissions import PermissionPolicy, is_risky_cli, is_risky_shell


def test_detects_risky_cli_commands():
    assert is_risky_cli(["league", "delete", "epl-2018"])
    assert is_risky_cli(["model", "delete", "epl-2018", "rf"])
    assert not is_risky_cli(["league", "list", "--catalog"])


def test_detects_risky_shell_commands():
    assert is_risky_shell(["rm", "-rf", "storage"])
    assert is_risky_shell(["git", "reset", "--hard"])
    assert not is_risky_shell(["git", "status"])


def test_permission_policy_blocks_risky_without_confirmation():
    policy = PermissionPolicy()

    decision = policy.check_shell("rm -rf storage")

    assert not decision.allowed
    assert decision.risky
    assert "confirmation" in decision.reason
