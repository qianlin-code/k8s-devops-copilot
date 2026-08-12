"""灌入演示账号的独立脚本。

密码仅从 COPILOT_SEED_PASSWORD 读取，脚本不会生成、回显或持久化明文密码。
"""

import os
import sys
from argparse import ArgumentParser

from app.storage.db import session_scope
from app.storage.seed import seed_test_users

if __name__ == "__main__":
    parser = ArgumentParser(description="创建或显式轮换隔离验收的演示账号密码")
    parser.add_argument(
        "--rotate-password",
        action="store_true",
        help="仅轮换固定 admin/demo-user 的密码；默认不改写已有用户",
    )
    args = parser.parse_args()
    password = os.environ.get("COPILOT_SEED_PASSWORD")
    if not password:
        print("缺少 COPILOT_SEED_PASSWORD，拒绝创建可猜测的演示账号。", file=sys.stderr)
        raise SystemExit(2)
    with session_scope() as session:
        result = seed_test_users(session, password=password, force=args.rotate_password)
    print(f"Seeded: {result}")
    print("Admin username='admin'; user username='demo-user'. Password was supplied by COPILOT_SEED_PASSWORD.")
