"""把 tool_call_audits 的幂等键从全局唯一改成 (conversation_id, request_id) 复合唯一。

背景: `request_id` 由 LLM 生成，实测会出现 "123456" 这类极易碰撞的值。原来
`UNIQUE(request_id)` 是全局的，导致两个问题：
  1. B 会话复用 A 会话用过的 request_id 时，`_find_replay` 会命中 A 的审计行并
     重放 A 的结果——写操作实际没执行却报告成功，而且串了别人的数据。
  2. 修复 `_find_replay` 按会话限定范围后，若表上仍是全局唯一约束，第二个会话
     会正常执行工具、然后在写审计时撞上唯一约束抛 IntegrityError（500）。
所以查询范围和表约束必须一起改，只改一边都不对。

SQLite 不支持 DROP/ALTER CONSTRAINT，只能新建表 -> 拷数据 -> 替换。全新环境
不需要跑这个脚本（`init_db()` 建表时就是新约束）。

**运行前请先备份** backend/data/app.db（连同同目录的 -wal/-shm，如果存在）。
本脚本会重建表，比 ADD COLUMN 风险高，务必先留快照。

运行: python scripts/migrate_audit_idempotency.py [--db-path path/to/app.db]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_OLD_CONSTRAINT = "uq_tool_call_request_id"
_NEW_CONSTRAINT = "uq_tool_call_conversation_request"

# 与 app/storage/models.py::ToolCallAudit 保持一致
_NEW_TABLE_DDL = f"""
CREATE TABLE tool_call_audits_new (
    id INTEGER NOT NULL,
    request_id VARCHAR(64),
    trace_id VARCHAR(64) NOT NULL,
    conversation_id VARCHAR(36),
    tool_name VARCHAR(64) NOT NULL,
    is_write BOOLEAN NOT NULL,
    arguments JSON NOT NULL,
    result JSON,
    success BOOLEAN NOT NULL,
    error_code VARCHAR(64),
    error_message TEXT,
    cache_hit BOOLEAN NOT NULL,
    idempotent_replay BOOLEAN NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT {_NEW_CONSTRAINT} UNIQUE (conversation_id, request_id)
)
"""

_COLUMNS = (
    "id, request_id, trace_id, conversation_id, tool_name, is_write, arguments, "
    "result, success, error_code, error_message, cache_hit, idempotent_replay, "
    "elapsed_ms, created_at"
)


def _table_sql(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tool_call_audits'"
    ).fetchone()
    return row[0] if row else None


def _find_collisions(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """找出会违反新复合约束的重复行。

    理论上不该有：旧约束是全局唯一，比新约束更严格，所以旧数据必然满足新约束。
    仍然检查一次——库可能被手工改过，重建表时撞约束会中途失败。
    """
    return conn.execute(
        "SELECT request_id, COUNT(*) FROM tool_call_audits "
        "WHERE request_id IS NOT NULL "
        "GROUP BY conversation_id, request_id HAVING COUNT(*) > 1"
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else ROOT / "data" / "app.db"
    if not db_path.exists():
        print(
            f"数据库文件不存在: {db_path}"
            "（新装环境不需要跑这个脚本，init_db() 建表时就是新约束）"
        )
        return 0

    print(f"目标数据库: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        current = _table_sql(conn)
        if current is None:
            print("tool_call_audits 表不存在，跳过（可能是全新的数据库）")
            return 0
        if _NEW_CONSTRAINT in current:
            print("约束已是 (conversation_id, request_id) 复合唯一，无需迁移")
            return 0
        if _OLD_CONSTRAINT not in current:
            print(
                "未找到预期的旧约束 uq_tool_call_request_id，表结构与预期不符。\n"
                "为避免误操作已中止，请人工核对后再处理。当前 DDL:\n"
                f"{current}"
            )
            return 1

        collisions = _find_collisions(conn)
        if collisions:
            print("存在会违反新约束的重复 (conversation_id, request_id)，已中止：")
            for request_id, count in collisions:
                print(f"  request_id={request_id!r} 重复 {count} 次")
            print("请先人工清理这些审计行，再重新运行。")
            return 1

        before = conn.execute("SELECT COUNT(*) FROM tool_call_audits").fetchone()[0]
        print(f"现有审计行数: {before}")
        print("!! 请确认已备份该文件（连同 -wal/-shm）：本脚本会重建表 !!")

        # 重建期间关外键约束检查，并整体放在一个事务里，中途失败自动回滚
        conn.execute("PRAGMA foreign_keys=OFF")
        with conn:
            conn.execute(_NEW_TABLE_DDL)
            conn.execute(
                f"INSERT INTO tool_call_audits_new ({_COLUMNS}) "
                f"SELECT {_COLUMNS} FROM tool_call_audits"
            )
            conn.execute("DROP TABLE tool_call_audits")
            conn.execute("ALTER TABLE tool_call_audits_new RENAME TO tool_call_audits")
            # 索引随旧表一起被 DROP，需要重建（名字与 models.py 一致）
            conn.execute(
                "CREATE INDEX ix_audit_tool_created "
                "ON tool_call_audits (tool_name, created_at)"
            )
            conn.execute(
                "CREATE INDEX ix_tool_call_audits_trace_id "
                "ON tool_call_audits (trace_id)"
            )
        conn.execute("PRAGMA foreign_keys=ON")

        after = conn.execute("SELECT COUNT(*) FROM tool_call_audits").fetchone()[0]
        if after != before:
            print(f"!! 行数不一致（迁移前 {before}，迁移后 {after}），请用备份核对 !!")
            return 1
        print(f"迁移完成：{after} 行已保留，约束改为 {_NEW_CONSTRAINT}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
