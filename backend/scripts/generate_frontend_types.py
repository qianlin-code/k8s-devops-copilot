"""从 FastAPI OpenAPI schema 生成前端 TypeScript 类型。

前端类型是生成物，禁止手写或手改。改动任何后端 schema 后必须重跑本脚本。
运行: python scripts/generate_frontend_types.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="typegen-"))
os.environ.setdefault("API_KEY", "typegen")
os.environ.setdefault("STARTUP_PROBE_EXTERNAL", "false")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'typegen.db').as_posix()}")
os.environ.setdefault("QDRANT_PATH", str(_TMP / "qdrant"))
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("LLM_PROVIDER", "ollama")

from app.main import create_app  # noqa: E402

OUTPUT = ROOT.parent / "frontend" / "src" / "api" / "types.ts"

HEADER = """/**
 * 本文件由 backend/scripts/generate_frontend_types.py 自动生成，请勿手动修改。
 * 后端修改任何接口 schema 后，重新运行该脚本以同步类型。
 *
 * Generated from OpenAPI schema of Enterprise Support Copilot.
 */

"""

_PRIMITIVES = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}


def ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def render_type(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return ref_name(schema["$ref"])
    if "anyOf" in schema or "oneOf" in schema:
        variants = schema.get("anyOf") or schema["oneOf"]
        rendered = sorted({render_type(v) for v in variants})
        return " | ".join(rendered)
    if "allOf" in schema:
        return " & ".join(render_type(v) for v in schema["allOf"])
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(v) for v in schema["enum"])

    kind = schema.get("type")
    if kind == "array":
        items = schema.get("items")
        return f"{render_type(items)}[]" if items else "unknown[]"
    if kind == "object":
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return f"Record<string, {render_type(extra)}>"
        return "Record<string, unknown>"
    if isinstance(kind, list):
        return " | ".join(_PRIMITIVES.get(k, "unknown") for k in kind)
    if kind in _PRIMITIVES:
        return _PRIMITIVES[kind]
    return "unknown"


def render_interface(name: str, schema: dict[str, Any]) -> str:
    if "enum" in schema and "properties" not in schema:
        variants = " | ".join(json.dumps(v) for v in schema["enum"])
        return f"export type {name} = {variants}\n"

    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [f"export interface {name} {{"]
    if description := schema.get("description"):
        lines.insert(0, f"/** {description.splitlines()[0]} */")
    for prop, prop_schema in properties.items():
        rendered = render_type(prop_schema)
        optional = "" if prop in required else "?"
        # Pydantic 的 Optional 字段会带 null 变体，转成 TS 的联合类型
        if doc := prop_schema.get("description"):
            lines.append(f"  /** {doc.splitlines()[0]} */")
        lines.append(f"  {prop}{optional}: {rendered}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_endpoint_map(paths: dict[str, Any]) -> str:
    rows: list[str] = ["export const API_ENDPOINTS = {"]
    for path, methods in sorted(paths.items()):
        for method, op in methods.items():
            operation_id = op.get("operationId", "")
            if not operation_id:
                continue
            key = operation_id.split("__")[0]
            rows.append(f"  {key}: {{ method: {method.upper()!r}, path: {path!r} }},")
    rows.append("} as const\n")
    return "\n".join(rows)


def main() -> int:
    spec = create_app().openapi()
    schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})

    blocks = [HEADER]
    for name in sorted(schemas):
        if name.startswith("HTTPValidationError") or name == "ValidationError":
            continue
        blocks.append(render_interface(name, schemas[name]))
    blocks.append(build_endpoint_map(spec.get("paths", {})))
    blocks.append(
        "export const API_KEY_HEADER = 'X-API-Key'\n"
        "export const TRACE_ID_HEADER = 'X-Trace-Id'\n"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"generated {OUTPUT.relative_to(ROOT.parent)}")
    print(f"  interfaces: {len(schemas)}")
    print(f"  endpoints: {sum(len(m) for m in spec.get('paths', {}).values())}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
