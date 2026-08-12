from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import MockDeployment, MockPod, Organization, User, UserRole

_BASE = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

# (namespace, name, phase, reason, node_name, restart_count, hours_ago)
# ops-demo/api-gateway-7f9c 是 Pod Pending 排查的主演示对象：资源不足导致调度失败
_PODS = [
    ("ops-demo", "api-gateway-7f9c", "Pending", "Unschedulable: Insufficient cpu", None, 0, 1),
    ("ops-demo", "worker-queue-2b1a", "Running", None, "node-1", 0, 6),
    ("ops-demo", "billing-sync-9d3e", "Running", "CrashLoopBackOff", "node-2", 12, 2),
    ("data-pipeline", "etl-loader-4c8f", "Pending", "ImagePullBackOff", None, 0, 1),
]

_DEPLOYMENTS = [
    ("ops-demo", "api-gateway", "registry.internal/api-gateway:v2.3.1", 3, 0),
    ("ops-demo", "worker-queue", "registry.internal/worker-queue:v1.8.0", 2, 2),
    ("ops-demo", "billing-sync", "registry.internal/billing-sync:v1.1.4", 1, 0),
    ("data-pipeline", "etl-loader", "registry.internal/etl-loader:v0.9.2", 2, 0),
]


def seed_mock_data(session: Session, *, force: bool = False) -> dict[str, int]:
    """灌入模拟 Pod/Deployment 状态。已有数据时默认跳过，保证可重复执行。"""
    existing = session.scalar(select(MockPod).limit(1))
    if existing is not None and not force:
        return {"pods": 0, "deployments": 0}

    pods = 0
    for namespace, name, phase, reason, node_name, restarts, hours_ago in _PODS:
        if session.get(MockPod, (namespace, name)) is not None:
            continue
        session.add(
            MockPod(
                namespace=namespace,
                name=name,
                phase=phase,
                reason=reason,
                node_name=node_name,
                restart_count=restarts,
                last_transition_at=_BASE - timedelta(hours=hours_ago),
            )
        )
        pods += 1

    deployments = 0
    for namespace, name, image, replicas, available in _DEPLOYMENTS:
        if session.get(MockDeployment, (namespace, name)) is not None:
            continue
        session.add(
            MockDeployment(
                namespace=namespace,
                name=name,
                image=image,
                replicas=replicas,
                available_replicas=available,
            )
        )
        deployments += 1

    session.flush()
    return {"pods": pods, "deployments": deployments}


def seed_test_users(
    session: Session, *, password: str, force: bool = False
) -> dict[str, int]:
    """灌入测试账号：1 个组织 + 2 个用户（admin/user）。

    密码必须由调用方显式传入。演示账号只应由独立初始化脚本创建，
    不能在应用启动或代码库里保留可猜测的默认密码。
    """
    org_id = "org-00000000-0000-0000-0000-000000000001"
    org = session.get(Organization, org_id)
    if org is not None and not force:
        return {"organizations": 0, "users": 0, "passwords_rotated": 0}

    organizations = 0
    if org is None:
        org = Organization(id=org_id, name="Demo Organization")
        session.add(org)
        organizations = 1

    # bcrypt work factor 12
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

    users_data = [
        ("user-admin", "admin", UserRole.ADMIN.value),
        ("user-regular", "demo-user", UserRole.USER.value),
    ]

    users = 0
    passwords_rotated = 0
    for user_id, username, role in users_data:
        user = session.get(User, user_id)
        if user is not None:
            if force:
                # 仅用于隔离验收现场恢复凭据；不改动身份、角色、组织或业务数据。
                user.password_hash = password_hash
                passwords_rotated += 1
            continue
        session.add(
            User(
                id=user_id,
                username=username,
                password_hash=password_hash,
                role=role,
                organization_id=org_id,
            )
        )
        users += 1

    session.flush()
    return {
        "organizations": organizations,
        "users": users,
        "passwords_rotated": passwords_rotated,
    }
