"""验收编排的证据文件不能泄露仅通过环境传递的运行时凭据。"""

import os
import sys
from pathlib import Path

from scripts.run_local_acceptance import Command, _run


def test_command_evidence_never_serializes_environment_credentials(tmp_path: Path) -> None:
    secret = "acceptance-runtime-secret-must-not-appear"
    log = tmp_path / "command.log"
    result = _run(
        Command(
            label="credential-redaction-probe",
            args=[sys.executable, "-c", "print('probe complete')"],
            cwd=Path.cwd(),
            env={**os.environ, "COPILOT_USER_PASSWORD": secret},
            log=log,
            timeout_seconds=10,
        )
    )

    assert result.passed
    content = log.read_text(encoding="utf-8")
    assert "runtime_credentials=environment-only" in content
    assert secret not in content
