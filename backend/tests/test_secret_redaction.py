import importlib.util
from pathlib import Path

from app.security.output_guard import sanitize_output


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "security_observability_check.py"
SPEC = importlib.util.spec_from_file_location("security_observability_check", SCRIPT)
assert SPEC and SPEC.loader
security_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(security_check)


def _fake_key() -> str:
    return "sk-" + "ws-" + "alpha.beta_gamma-delta0123456789"


def test_dotted_workspace_key_is_redacted_from_model_output() -> None:
    secret = _fake_key()
    result = sanitize_output(f"credential={secret}")

    assert secret not in result.text
    assert "credential_or_path" in result.redactions


def test_secret_scan_reports_location_and_kind_without_value(tmp_path: Path) -> None:
    secret = _fake_key()
    source = tmp_path / "probe.txt"
    source.write_text(f"prefix {secret} suffix\n", encoding="utf-8")

    count, findings = security_check._scan([source])

    assert count == 1
    assert len(findings) == 1
    assert findings[0]["line"] == 1
    assert findings[0]["kind"] == "api_key"
    assert findings[0]["value"] == "[REDACTED]"
    assert secret not in str(findings)


def test_deepseek_key_assignment_is_redacted_without_retaining_value() -> None:
    secret = _fake_key()
    result = sanitize_output(f"DEEPSEEK_API_KEY={secret}")

    assert secret not in result.text
    assert "credential_or_path" in result.redactions
