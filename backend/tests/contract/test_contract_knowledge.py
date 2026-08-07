from fastapi.testclient import TestClient

from app.schemas.knowledge import (
    DeleteDocumentResponse,
    DocumentListResponse,
    IngestResponse,
    SedimentationEntry,
    SedimentationListResponse,
)
from tests.conftest import API_HEADERS
from tests.contract.test_contract_basics import assert_error_contract
from tests.fakes import ScriptedLLMClient

DOC = (
    "# 缓存刷新指引\n\n"
    "## 何时需要刷新\n"
    "管理员调整权限后，需要刷新权限缓存才能生效。\n"
)


def test_ingest_matches_schema(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/knowledge/documents",
        headers=API_HEADERS,
        json={"title": "缓存刷新指引", "content": DOC, "chunk_strategy": "markdown"},
    )
    assert resp.status_code == 200, resp.text
    body = IngestResponse.model_validate(resp.json())
    assert body.document.chunk_count >= 1
    assert body.document.source == "upload"
    assert body.document.chunk_strategy == "markdown"
    assert body.bm25_index_size == body.document.chunk_count


def test_ingest_rejects_blank_content(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/knowledge/documents",
        headers=API_HEADERS,
        json={"title": "空文档", "content": "   "},
    )
    assert resp.status_code == 422
    assert assert_error_contract(resp.json()).code == "VALIDATION_FAILED"


def test_ingest_rejects_unknown_strategy(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/knowledge/documents",
        headers=API_HEADERS,
        json={"title": "文档", "content": DOC, "chunk_strategy": "quantum"},
    )
    assert resp.status_code == 400
    error = assert_error_contract(resp.json())
    assert error.code == "VALIDATION_FAILED"
    assert error.details["chunk_strategy"] == "quantum"


def test_list_documents_matches_schema(client: TestClient, seeded_kb: str) -> None:
    resp = client.get("/api/v1/knowledge/documents", headers=API_HEADERS)
    assert resp.status_code == 200
    body = DocumentListResponse.model_validate(resp.json())
    assert body.total == 1
    assert body.vector_count == body.bm25_index_size
    assert body.collection_name.startswith("kb_")


def test_delete_document_matches_schema(client: TestClient, seeded_kb: str) -> None:
    resp = client.delete(
        f"/api/v1/knowledge/documents/{seeded_kb}", headers=API_HEADERS
    )
    assert resp.status_code == 200
    body = DeleteDocumentResponse.model_validate(resp.json())
    assert body.deleted is True
    assert body.vector_count == 0
    assert body.bm25_index_size == 0


def test_delete_unknown_document(client: TestClient) -> None:
    resp = client.delete("/api/v1/knowledge/documents/nope", headers=API_HEADERS)
    assert resp.status_code == 404
    assert assert_error_contract(resp.json()).code == "RESOURCE_NOT_FOUND"


def _make_conversation(client: TestClient, llm: ScriptedLLMClient) -> str:
    llm.queue(
        {"action": "answer", "reasoning": "可直接回答", "confidence": 0.9},
        {"sufficient": True, "reasoning": "片段足够"},
        "permission_level 为 restricted 会导致 403，请提权到 standard[1]。",
    )
    resp = client.post(
        "/api/v1/chat",
        headers=API_HEADERS,
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["conversation_id"]


def test_sedimentation_requires_human_approval(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    conversation_id = _make_conversation(client, llm)
    before = client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()["total"]

    marked = client.post(
        "/api/v1/knowledge/sedimentations",
        headers=API_HEADERS,
        json={
            "conversation_id": conversation_id,
            "marked_by": "admin",
            "proposed_title": "403 登录失败处理",
        },
    )
    assert marked.status_code == 200, marked.text
    entry = SedimentationEntry.model_validate(marked.json())
    assert entry.status == "pending"
    assert entry.kb_document_id is None

    # 标记本身绝不入库
    mid = client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()["total"]
    assert mid == before

    approved = client.post(
        f"/api/v1/knowledge/sedimentations/{entry.pending_id}/review",
        headers=API_HEADERS,
        json={"reviewer": "admin", "approved": True, "note": "内容准确"},
    )
    assert approved.status_code == 200, approved.text
    reviewed = SedimentationEntry.model_validate(approved.json())
    assert reviewed.status == "approved"
    assert reviewed.kb_document_id is not None
    assert reviewed.reviewed_at is not None

    after = client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()
    assert after["total"] == before + 1
    assert any(d["source"] == "sedimentation" for d in after["documents"])


def test_sedimentation_rejection_does_not_ingest(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    conversation_id = _make_conversation(client, llm)
    before = client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()["total"]

    entry = client.post(
        "/api/v1/knowledge/sedimentations",
        headers=API_HEADERS,
        json={"conversation_id": conversation_id, "marked_by": "admin"},
    ).json()

    rejected = client.post(
        f"/api/v1/knowledge/sedimentations/{entry['pending_id']}/review",
        headers=API_HEADERS,
        json={"reviewer": "admin", "approved": False, "note": "质量不足"},
    )
    assert rejected.status_code == 200
    assert SedimentationEntry.model_validate(rejected.json()).status == "rejected"
    assert (
        client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()["total"]
        == before
    )


def test_duplicate_mark_rejected(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    conversation_id = _make_conversation(client, llm)
    payload = {"conversation_id": conversation_id, "marked_by": "admin"}
    client.post("/api/v1/knowledge/sedimentations", headers=API_HEADERS, json=payload)
    second = client.post(
        "/api/v1/knowledge/sedimentations", headers=API_HEADERS, json=payload
    )
    assert second.status_code == 400
    assert assert_error_contract(second.json()).code == "BUSINESS_RULE_VIOLATION"


def test_double_review_rejected(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    conversation_id = _make_conversation(client, llm)
    entry = client.post(
        "/api/v1/knowledge/sedimentations",
        headers=API_HEADERS,
        json={"conversation_id": conversation_id, "marked_by": "admin"},
    ).json()
    url = f"/api/v1/knowledge/sedimentations/{entry['pending_id']}/review"
    body = {"reviewer": "admin", "approved": True}
    assert client.post(url, headers=API_HEADERS, json=body).status_code == 200
    second = client.post(url, headers=API_HEADERS, json=body)
    assert second.status_code == 400
    assert assert_error_contract(second.json()).code == "BUSINESS_RULE_VIOLATION"


def test_mark_unknown_conversation(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/knowledge/sedimentations",
        headers=API_HEADERS,
        json={"conversation_id": "missing", "marked_by": "admin"},
    )
    assert resp.status_code == 404
    assert assert_error_contract(resp.json()).code == "RESOURCE_NOT_FOUND"


def test_list_sedimentations_matches_schema(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    conversation_id = _make_conversation(client, llm)
    client.post(
        "/api/v1/knowledge/sedimentations",
        headers=API_HEADERS,
        json={"conversation_id": conversation_id, "marked_by": "admin"},
    )
    resp = client.get(
        "/api/v1/knowledge/sedimentations", headers=API_HEADERS, params={"status": "pending"}
    )
    assert resp.status_code == 200
    body = SedimentationListResponse.model_validate(resp.json())
    assert body.total == 1
    assert body.entries[0].status == "pending"
