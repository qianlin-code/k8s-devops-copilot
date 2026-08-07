from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import MockAccount, MockOrder

_BASE = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

_ACCOUNTS = [
    # u-1001 是 403 排查的主演示账号：权限不足导致登录被拒
    ("u-1001", "li.wei@acme-corp.com", "active", None, "restricted", 2),
    ("u-1002", "zhang.min@acme-corp.com", "locked", "连续 5 次密码错误触发风控锁定", "standard", 1),
    ("u-1003", "chen.hao@acme-corp.com", "suspended", "账期逾期，服务暂停", "standard", 1),
    ("u-1004", "wang.fang@acme-corp.com", "active", None, "admin", 3),
]

_ORDERS = [
    ("ORD-20260731-001", "u-1001", "Copilot 企业版 年度订阅", "paid", 12800.0, 7),
    ("ORD-20260802-014", "u-1001", "席位扩容 x20", "pending_payment", 4200.0, 5),
    ("ORD-20260715-233", "u-1003", "Copilot 标准版 年度订阅", "overdue", 6800.0, 23),
    ("ORD-20260805-077", "u-1004", "私有化部署支持包", "paid", 45000.0, 2),
]


def seed_mock_data(session: Session, *, force: bool = False) -> dict[str, int]:
    """灌入模拟账号/订单。已有数据时默认跳过，保证可重复执行。"""
    existing = session.scalar(select(MockAccount).limit(1))
    if existing is not None and not force:
        return {"accounts": 0, "orders": 0}

    accounts = 0
    for user_id, email, status, reason, level, cache_version in _ACCOUNTS:
        if session.get(MockAccount, user_id) is not None:
            continue
        session.add(
            MockAccount(
                user_id=user_id,
                email=email,
                status=status,
                locked_reason=reason,
                permission_level=level,
                last_login_at=_BASE - timedelta(hours=cache_version * 3),
                cache_version=cache_version,
            )
        )
        accounts += 1

    orders = 0
    for order_id, user_id, product, status, amount, days_ago in _ORDERS:
        if session.get(MockOrder, order_id) is not None:
            continue
        session.add(
            MockOrder(
                id=order_id,
                user_id=user_id,
                product=product,
                status=status,
                amount=amount,
                created_at=_BASE - timedelta(days=days_ago),
            )
        )
        orders += 1

    session.flush()
    return {"accounts": accounts, "orders": orders}
