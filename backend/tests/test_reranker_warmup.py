"""Rerank 模型加载策略测试。

两个实测踩到的坑：
1. 模型在本地也会联网校验版本，国内网络下重试 5 次让首个请求白等 300s
2. 不预热的话冷启动开销全落在第一个用户请求上
"""

import os
from pathlib import Path

import pytest

from app.rag.reranker import (
    BGEReranker,
    IdentityReranker,
    _enable_hf_offline_if_cached,
    preload_reranker,
    set_reranker,
)


_OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in _OFFLINE_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_offline_enabled_when_model_cached(monkeypatch, tmp_path: Path) -> None:
    """本地已有模型缓存时切成离线，避免联网校验白等 300s。"""
    (tmp_path / "hub" / "models--BAAI--bge-reranker-base").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for key in _OFFLINE_KEYS:
        os.environ.pop(key, None)

    assert _enable_hf_offline_if_cached() is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_not_forced_when_cache_missing(monkeypatch, tmp_path: Path) -> None:
    """缓存不存在时不能强制离线，否则首次下载会被挡住。"""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for key in _OFFLINE_KEYS:
        os.environ.pop(key, None)

    assert _enable_hf_offline_if_cached() is False
    assert "HF_HUB_OFFLINE" not in os.environ


def test_existing_offline_setting_is_respected(monkeypatch, tmp_path: Path) -> None:
    """用户显式设过就不覆盖。"""
    (tmp_path / "hub" / "models--BAAI--bge-reranker-base").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    assert _enable_hf_offline_if_cached() is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_prefer_local_files_patches_imported_modules(
    monkeypatch, tmp_path: Path
) -> None:
    """只设环境变量不够：已 import 的 HF 模块把它固化成常量了。

    这是实测撞出来的——环境变量设对了，但预热仍白等 294s。
    """
    import sys
    import types

    (tmp_path / "hub" / "models--BAAI--bge-reranker-base").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for key in _OFFLINE_KEYS:
        os.environ.pop(key, None)

    fake = types.ModuleType("huggingface_hub.constants")
    fake.HF_HUB_OFFLINE = False  # 模拟 import 时读到的旧值
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)

    BGEReranker._prefer_local_files()
    assert fake.HF_HUB_OFFLINE is True, "已 import 的模块常量也要改"


def test_preload_is_noop_for_stub_reranker() -> None:
    """替身 reranker 没有 _ensure_model，预热不能因此报错。"""
    set_reranker(IdentityReranker())
    preload_reranker()  # 不应抛异常


def test_preload_invokes_model_loading() -> None:
    calls: list[int] = []

    class Probe(IdentityReranker):
        def _ensure_model(self) -> None:
            calls.append(1)

    set_reranker(Probe())
    preload_reranker()
    assert calls == [1]


def test_cpu_device_disables_fp16() -> None:
    """CPU 上 fp16 会显著变慢甚至不支持。"""
    reranker = BGEReranker("stub", use_fp16=True, device="cpu")
    assert reranker._resolve_device() == "cpu"
