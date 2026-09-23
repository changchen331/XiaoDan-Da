"""知识库：切分 / 清洗 / 索引重建幂等 / 附件元数据 sidecar（milvus 侧用假客户端验证）。"""

import json
from pathlib import Path

import pytest

from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.cleaner import clean_text


# ==================== 切分策略 ====================


def test_chunker_faq_pairs() -> None:
    """FAQ 切分：按问答对边界切，每对一个 chunk。"""
    text = "问：校园卡丢了怎么办？\n答：请到一卡通中心挂失补办。\n问：图书馆几点开门？\n答：早8点。"
    chunks = chunk_by_type(text, "FAQ")
    assert len(chunks) == 2
    assert "校园卡" in chunks[0]["text"]
    assert chunks[0]["metadata"]["doc_type"] == "FAQ"


def test_chunker_notice_by_title() -> None:
    """通知切分：按标题行分界，每条通知一个 chunk。"""
    text = (
        "关于2026年选课安排的通知\n第一条内容。\n\n" "关于宿舍调整的通知\n第二条内容。"
    )
    chunks = chunk_by_type(text, "通知")
    assert len(chunks) == 2


def test_chunker_manual_splits_long_text() -> None:
    """手册切分：长文本被递归切为多个 chunk 且带重叠。"""
    long_text = "\n\n".join(
        f"第{i}段落内容。" + "课程设置的详细说明。" * 30 for i in range(20)
    )
    chunks = chunk_by_type(long_text, "手册")
    assert len(chunks) > 1
    assert all(chunk["metadata"]["doc_type"] == "手册" for chunk in chunks)


# ==================== 文本清洗 ====================


def test_cleaner_removes_noise() -> None:
    """清洗：页脚版权、备案号、分享残留等噪声行被去除。"""
    text = "复旦大学选课须知\n正文第一段。\n版权所有 © 复旦大学\n沪ICP备00000001号\n正文第二段。"
    cleaned = clean_text(text)
    assert "版权所有" not in cleaned
    assert "沪ICP" not in cleaned
    assert "正文第一段" in cleaned


# ==================== 索引重建幂等（v2-plan 2.1 #7） ====================


def test_delete_chunks_by_source_uses_metadata_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按来源删除：filter 必须精确指向 metadata["source_file"（相对路径）]。"""
    from config.settings import settings
    from knowledge_base.indexing import milvus_client

    captured: dict = {}

    class _FakeClient:
        def delete(self, collection_name: str, filter: str) -> dict:
            captured["collection_name"] = collection_name
            captured["filter"] = filter
            return {"delete_count": 3}

    monkeypatch.setattr(milvus_client, "get_milvus_client", lambda: _FakeClient())

    deleted = milvus_client.delete_chunks_by_source("手册/选课手册.pdf")

    assert deleted == 3
    assert captured["collection_name"] == settings.MILVUS_COLLECTION
    assert captured["filter"] == 'metadata["source_file"] == "手册/选课手册.pdf"'


def test_faq_build_clears_before_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAQ 重建：先清空既有行再写入，重跑不累积重复问答。

    注：knowledge_base.faq 顶层会导入 embeddings（连带 FlagEmbedding，
    实测约 11s），故先用 sys.modules 注入假 embeddings 再导入该模块。
    """
    import importlib
    import sys
    import types

    fake_embeddings = types.ModuleType("knowledge_base.indexing.embeddings")
    fake_embeddings.get_embedder = lambda: None
    monkeypatch.setitem(
        sys.modules, "knowledge_base.indexing.embeddings", fake_embeddings
    )
    faq_module = importlib.import_module("knowledge_base.faq")

    calls: list = []

    class _FakeClient:
        def delete(self, collection_name: str, filter: str) -> dict:
            calls.append(("delete", filter))
            return {"delete_count": 0}

    class _FakeEmbedder:
        def encode(self, texts: list) -> list:
            return [{"dense": [0.0], "sparse": {}} for _ in texts]

    monkeypatch.setattr(faq_module, "get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(faq_module, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(
        faq_module, "ensure_collection_loaded", lambda client, name: None
    )

    def _fake_insert(rows: list) -> int:
        calls.append(("insert", len(rows)))
        return len(rows)

    monkeypatch.setattr(faq_module, "insert_faq_rows", _fake_insert)

    index = faq_module.FAQIndex()
    written = index.build(
        [
            {
                "question": "校园卡丢了怎么办？",
                "answer": "请到一卡通中心挂失补办。",
                "similar_questions": ["校园卡挂失"],
                "tags": ["校园卡"],
            }
        ]
    )

    # 顺序契约：先清空（否则重跑会把整套问答再插一遍）
    assert calls[0] == ("delete", "id >= 0")
    # 标准问题 + 1 条相似问法各占一行
    assert calls[1] == ("insert", 2)
    assert written == 2


# ==================== 附件元数据 sidecar（v2-plan 2.1 #12） ====================


def test_crawler_writes_sidecar_for_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """爬虫下载附件时必须旁挂元数据文件：附件正文没有头部可解析，
    不旁挂则人群 / 时效过滤对附件永远失效。"""
    from knowledge_base.crawler import fudan_crawler

    crawler = fudan_crawler.FudanCrawler(rate_limit=0.0)

    class _FakeResponse:
        content = b"x" * 4096

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(crawler.session, "get", lambda url, timeout=60: _FakeResponse())
    monkeypatch.setattr(fudan_crawler, "ATTACHMENT_OUTPUT_DIR", str(tmp_path))

    sidecar_meta = {
        "标题": "关于2026年春季学期选课安排的通知",
        "发布日期": "2026-01-15",
        "原文链接": "https://jwc.fudan.edu.cn/xx/xxxx.htm",
        "适用对象": "本科生",
        "生效学期": "2026-2027秋季",
    }
    # noinspection PyProtectedMember —— 附件下载带网络副作用，不宜为测试公开成 API
    file_path = crawler._download_attachment(
        "https://jwc.fudan.edu.cn/files/notice.pdf",
        "关于2026年春季学期选课安排的通知",
        sidecar_meta,
    )

    assert file_path is not None
    sidecar_path = Path(file_path + ".meta.json")
    assert sidecar_path.exists()
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert saved["适用对象"] == "本科生"
    assert saved["生效学期"] == "2026-2027秋季"


def test_corpus_metadata_priority_header_over_sidecar(tmp_path: Path) -> None:
    """元数据三档优先级：文档自带头部 > sidecar > 默认值。"""
    from knowledge_base.preprocessing.corpus import load_document

    notice_dir = tmp_path / "通知"
    notice_dir.mkdir(parents=True)

    # 情形一：无头部 + 有 sidecar（附件）→ 采用 sidecar，过滤条件真正生效
    attachment_like = notice_dir / "附件通知.txt"
    attachment_like.write_text("正文内容。", encoding="utf-8")
    (notice_dir / "附件通知.txt.meta.json").write_text(
        json.dumps(
            {
                "标题": "附件通知标题",
                "原文链接": "https://jwc.fudan.edu.cn/xx/yyyy.htm",
                "适用对象": "研究生",
                "生效学期": "2025-2026春季",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    from_sidecar = load_document(str(attachment_like), "通知", "通知")
    assert from_sidecar["metadata"]["target_audience"] == "研究生"
    assert from_sidecar["metadata"]["valid_semester"] == "2025-2026春季"
    assert (
        from_sidecar["metadata"]["source_url"]
        == "https://jwc.fudan.edu.cn/xx/yyyy.htm"
    )
    assert from_sidecar["metadata"]["title"] == "附件通知标题"

    # 情形二：头部与 sidecar 同时存在 → 文档自带头部优先
    with_header = notice_dir / "带头部.txt"
    with_header.write_text(
        "【标题】头部标题\n【适用对象】本科生\n\n正文内容。", encoding="utf-8"
    )
    (notice_dir / "带头部.txt.meta.json").write_text(
        json.dumps({"适用对象": "研究生"}, ensure_ascii=False), encoding="utf-8"
    )
    from_header = load_document(str(with_header), "通知", "通知")
    assert from_header["metadata"]["target_audience"] == "本科生"


# ==================== 大 PDF 解析可观测性（v2-plan 2.1 #5） ====================


def test_parse_file_task_reports_elapsed(tmp_path: Path) -> None:
    """单文件解析耗时随结果返回：主进程靠它汇总"最慢文件"（2.1 #5）。"""
    from knowledge_base.preprocessing.corpus import _parse_file_task

    notice = tmp_path / "通知" / "x.txt"
    notice.parent.mkdir(parents=True)
    notice.write_text("正文内容。", encoding="utf-8")

    # noinspection PyProtectedMember —— 耗时字段是任务函数的内部契约，
    # 无公开入口可直接断言；此处直接调用以钉住"必须返回 elapsed"
    cache_key, parsed = _parse_file_task(
        ("通知/x.txt", str(notice), {"size": 12, "mtime": 0}, "通知", "通知")
    )

    assert cache_key == "通知/x.txt"
    assert "error" not in parsed
    assert parsed["elapsed"] >= 0.0
