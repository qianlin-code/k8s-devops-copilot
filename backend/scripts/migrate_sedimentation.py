"""为 pending_sedimentations 表补齐自动质量初筛字段的一次性迁移。

背景: 项目没有引入 Alembic（体量还没到需要迁移框架的程度），SQLite 上最简单的
升级方式是删库重建，但那样会丢掉本地已经积累的对话/工单/审计数据。
这个脚本用 SQLite 原生的 ALTER TABLE ADD COLUMN（不支持删列/改列，但加列足够用）
就地补齐新字段，已存在的列会被跳过，可重复执行。

**运行前请先备份** backend/data/app.db（以及同目录的 -wal/-shm 文件，如果存在）。
本脚本只做加列，不会删除现有数据，但任何直接改库结构的操作都建议先留一份快照。

运行: python scripts/migrate_sedimentation.py [--db-path path/to/app.db]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (列名, SQLite 类型, 默认值表达式)
_NEW_COLUMNS = [
    ("quality_score", "REAL", "NULL"),
    ("quality_reasoning", "TEXT", "NULL"),
    ("duplicate_of_document_id", "VARCHAR(36)", "NULL"),
    ("duplicate_score", "REAL", "NULL"),
    ("reviewed_by", "VARCHAR(128)", "NULL"),
    ("auto_approved", "BOOLEAN", "0"),
]


def _default_db_path() -> Path:
    return ROOT / "data" / "app.db"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else _default_db_path()
    if not db_path.exists():
        print(f"数据库文件不存在: {db_path}（新装环境不需要跑这个脚本，init_db() 会直接建出新字段）")
        return 0

    print(f"目标数据库: {db_path}")
    print("!! 请确认已经备份该文件（以及同目录的 -wal/-shm），本脚本仅做 ADD COLUMN 但操作前务必留快照 !!")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("PRAGMA table_info(pending_sedimentations)")
        existing = {row[1] for row in cur.fetchall()}
        if not existing:
            print("pending_sedimentations 表不存在，跳过（可能是全新的数据库）")
            return 0

        added = []
        for name, sqltype, default in _NEW_COLUMNS:
            if name in existing:
                continue
            conn.execute(
                f"ALTER TABLE pending_sedimentations ADD COLUMN {name} {sqltype} DEFAULT {default}"
            )
            added.append(name)
        conn.commit()

        if added:
            print(f"已添加字段: {', '.join(added)}")
        else:
            print("字段已是最新，无需迁移")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
