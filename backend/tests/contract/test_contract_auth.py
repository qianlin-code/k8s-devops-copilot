from fastapi.testclient import TestClient

from app.schemas.auth import LoginResponse, RegisterResponse
from tests.contract.test_contract_basics import assert_error_contract


def test_register_then_login_issues_jwt(client: TestClient) -> None:
    registered = client.post(
        "/api/v1/auth/register",
        json={
            "username": "frontend-e2e-user",
            "password": "frontend-e2e-password",
            "organization_name": "Frontend E2E Organization",
        },
    )
    assert registered.status_code == 200, registered.text
    account = RegisterResponse.model_validate(registered.json())
    assert account.role == "user"

    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": "frontend-e2e-user", "password": "frontend-e2e-password"},
    )
    assert logged_in.status_code == 200, logged_in.text
    session = LoginResponse.model_validate(logged_in.json())
    assert session.user_id == account.user_id
    assert session.organization_id == account.organization_id
    assert session.access_token

    conversations = client.get(
        "/api/v1/conversations",
        headers={"Authorization": f"Bearer {session.access_token}"},
    )
    assert conversations.status_code == 200
    assert conversations.json()["total"] == 0


def test_login_rejects_invalid_credentials(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "missing-user", "password": "incorrect-password"},
    )
    assert response.status_code == 401
    assert assert_error_contract(response.json()).code == "UNAUTHORIZED"
