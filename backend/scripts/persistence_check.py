"""Verify data remains readable after an existing acceptance stack recovers.

Credentials are accepted only through the process environment. The report
contains counts and HTTP statuses, never access tokens, usernames, passwords,
conversation content, or audit payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE = os.environ.get("COPILOT_BASE", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _login(client: httpx.Client, username: str, password: str) -> dict[str, str]:
    response = client.post(f"{BASE}/auth/login", json={"username": username, "password": password})
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _count(body: object, key: str) -> int | None:
    return body.get(key) if isinstance(body, dict) and isinstance(body.get(key), int) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="recovery persistence read check")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)) as client:
            admin_headers = _login(client, _required("COPILOT_ADMIN_USERNAME"), _required("COPILOT_ADMIN_PASSWORD"))
            user_headers = _login(client, _required("COPILOT_USER_USERNAME"), _required("COPILOT_USER_PASSWORD"))
            for label, path, headers, count_key, minimum in (
                ("knowledge_documents", "/knowledge/documents?limit=100", admin_headers, "total", 7),
                ("conversation_history", "/conversations?limit=100", user_headers, "total", 1),
                ("tool_audits", "/tool-audits?limit=100", user_headers, "total", 1),
            ):
                response = client.get(f"{BASE}{path}", headers=headers)
                body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                observed = _count(body, count_key)
                checks.append(
                    {
                        "label": label,
                        "http_status": response.status_code,
                        "count": observed,
                        "minimum": minimum,
                        "passed": response.status_code == 200 and observed is not None and observed >= minimum,
                    }
                )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        checks.append({"label": "authentication_or_read", "error_type": type(exc).__name__, "passed": False})

    payload = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_url": BASE,
        "recovery_event": "host reboot; compose containers automatically recovered",
        "runtime_credentials": "environment-only-not-serialized",
        "checks": checks,
        "passed": bool(checks) and all(item.get("passed") is True for item in checks),
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "passed": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
