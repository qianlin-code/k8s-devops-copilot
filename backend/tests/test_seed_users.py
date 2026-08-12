"""隔离验收演示账号初始化的回归测试。"""

import bcrypt

from app.storage.db import session_scope
from app.storage.models import User
from app.storage.seed import seed_test_users


def test_seed_test_users_does_not_rotate_existing_password_without_force():
    with session_scope() as session:
        before = session.get(User, "user-admin")
        assert before is not None
        original_hash = before.password_hash

        result = seed_test_users(session, password="different-test-password")

    assert result == {"organizations": 0, "users": 0, "passwords_rotated": 0}
    with session_scope() as session:
        after = session.get(User, "user-admin")
        assert after is not None
        assert after.password_hash == original_hash


def test_seed_test_users_rotates_only_fixed_demo_account_passwords_when_forced():
    rotated_password = "rotated-isolated-acceptance-password"

    with session_scope() as session:
        result = seed_test_users(session, password=rotated_password, force=True)
        admin = session.get(User, "user-admin")
        user = session.get(User, "user-regular")
        assert admin is not None
        assert user is not None
        assert bcrypt.checkpw(rotated_password.encode(), admin.password_hash.encode())
        assert bcrypt.checkpw(rotated_password.encode(), user.password_hash.encode())

    assert result == {"organizations": 0, "users": 0, "passwords_rotated": 2}
