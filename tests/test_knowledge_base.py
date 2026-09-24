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


# ==================== 索引重建幂等 ====================


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


# ==================== Milvus 客户端复用 ====================


def test_milvus_client_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """客户端必须是进程级单例：建库流程会调用 6 次，每次新建等于反复重建连接。

    此前 `get_milvus_client()` 是工厂（每次 `MilvusClient(uri=...)`），
    而 docstring 写"可重复创建"——注释读起来像"已复用"，实现却是每次新实例。
    """
    from knowledge_base.indexing import milvus_client

    built: list = []

    class _FakeClient:
        def __init__(self, uri: str) -> None:
            built.append(uri)

    monkeypatch.setattr(milvus_client, "MilvusClient", _FakeClient)
    monkeypatch.setattr(milvus_client, "_client", None)

    first = milvus_client.get_milvus_client()
    second = milvus_client.get_milvus_client()

    assert first is second
    assert len(built) == 1  # 只构造一次


# ==================== 建库流水线 ====================


class _FakeEmbedder:
    """替身编码器：只回固定长度的假向量，不加载 BGE-M3。"""

    def encode(self, texts: list) -> list:
        return [{"dense": [0.0], "sparse": {}} for _ in texts]


def _import_build_module(monkeypatch: pytest.MonkeyPatch):
    """导入建库流水线模块。

    先注入假 embeddings：**本模块（build_index）顶层**会导入 embeddings
    （连带 FlagEmbedding，实测约 11s），单测不该为它付加载代价。
    （FAQ 侧的重型导入已下沉到 `FAQIndex.__init__`，导入 `knowledge_base.faq`
    不再触发加载，故这里只为 build_index 自己注入。）
    """
    import importlib
    import sys
    import types

    fake_embeddings = types.ModuleType("knowledge_base.indexing.embeddings")
    fake_embeddings.get_embedder = lambda: _FakeEmbedder()
    monkeypatch.setitem(
        sys.modules, "knowledge_base.indexing.embeddings", fake_embeddings
    )
    return importlib.import_module("knowledge_base.indexing.build_index")


def test_build_index_pipeline_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """流水线的步骤顺序即契约：建集合 → FAQ → 按来源幂等替换 → 分组入库。

    这条流水线此前只存在于 scripts/build_index.py，既无单测覆盖、也无法被
    第二个入口复用；下沉到模块层后用它钉住顺序与统计口径。
    """
    build_module = _import_build_module(monkeypatch)
    calls: list = []

    class _FakeFaqIndex:
        def build(self, entries: list) -> int:
            calls.append(("faq", len(entries)))
            return 5

    def _fake_delete(source: str) -> int:
        calls.append(("delete", source))
        return 2

    def _fake_chunk(text: str, doc_type: str, metadata: dict) -> list:
        return [{"text": f"{text}#1"}, {"text": f"{text}#2"}]

    records = [
        {
            "text": "甲正文",
            "doc_type": "通知",
            "topic_tag": "通知",
            "metadata": {"source_file": "通知/a.txt"},
        },
        {
            "text": "乙正文",
            "doc_type": "手册",
            "topic_tag": "手册",
            "metadata": {"source_file": "手册/b.pdf"},
        },
    ]
    failed = [("bad.pdf", "解析失败")]

    monkeypatch.setattr(
        build_module, "create_kb_collection", lambda: calls.append("create_kb")
    )
    monkeypatch.setattr(
        build_module, "create_faq_collection", lambda: calls.append("create_faq")
    )
    monkeypatch.setattr(
        build_module, "reset_collections", lambda: calls.append("reset")
    )
    monkeypatch.setattr(build_module, "get_faq_index", lambda: _FakeFaqIndex())
    monkeypatch.setattr(build_module, "delete_chunks_by_source", _fake_delete)
    monkeypatch.setattr(build_module, "chunk_by_type", _fake_chunk)
    monkeypatch.setattr(build_module, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(
        build_module, "iter_document_records", lambda data_dir: (records, failed)
    )
    monkeypatch.setattr(
        build_module,
        "insert_chunks",
        lambda chunks, vectors: calls.append(("insert", len(chunks))),
    )

    (tmp_path / "faq.json").write_text(
        '[{"question": "q", "answer": "a"}]', encoding="utf-8"
    )

    stats = build_module.build_index(str(tmp_path))

    # 统计口径（供第二入口消费）
    assert stats["chunk_count"] == 4  # 两组 × 每篇 2 个 chunk
    assert stats["faq_rows"] == 5
    assert stats["failed_files"] == failed
    # 顺序：集合就绪 → FAQ → 按来源替换 → 分组入库
    assert calls == [
        "create_kb",
        "create_faq",
        ("faq", 1),
        ("delete", "通知/a.txt"),
        ("delete", "手册/b.pdf"),
        ("insert", 2),
        ("insert", 2),
    ]


def test_build_index_rebuild_skips_per_source_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """整库重建：先清空集合，不再逐篇按来源删（避免对空集合发无谓请求）。"""
    build_module = _import_build_module(monkeypatch)
    calls: list = []

    monkeypatch.setattr(
        build_module, "create_kb_collection", lambda: calls.append("create_kb")
    )
    monkeypatch.setattr(
        build_module, "create_faq_collection", lambda: calls.append("create_faq")
    )
    monkeypatch.setattr(
        build_module, "reset_collections", lambda: calls.append("reset")
    )
    monkeypatch.setattr(
        build_module, "delete_chunks_by_source", lambda source: calls.append("delete")
    )
    monkeypatch.setattr(
        build_module, "chunk_by_type", lambda text, doc_type, metadata: []
    )
    monkeypatch.setattr(build_module, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(
        build_module, "iter_document_records", lambda data_dir: ([], [])
    )

    stats = build_module.build_index(str(tmp_path), rebuild=True)

    assert stats["chunk_count"] == 0
    assert stats["faq_rows"] == 0  # 无 faq.json
    assert calls == ["reset", "create_kb", "create_faq"]  # 无 delete


# ==================== FAQ 重建 ====================


def test_faq_build_clears_before_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAQ 重建：先清空既有行再写入，重跑不累积重复问答。

    注：`FAQIndex` 的重型依赖（embeddings / Milvus）是**实例化时才导入**的，
    故这里替换的是它们所在模块的属性，而不再替换 `knowledge_base.faq` 的模块级符号；
    embeddings 仍用 sys.modules 假模块顶掉，免得单测为 FlagEmbedding（约 11s）付代价。
    """
    import sys
    import types

    calls: list = []

    class _FakeClient:
        def delete(self, collection_name: str, filter: str) -> dict:
            calls.append(("delete", filter))
            return {"delete_count": 0}

    class _FakeEmbedder:
        def encode(self, texts: list) -> list:
            return [{"dense": [0.0], "sparse": {}} for _ in texts]

    fake_embeddings = types.ModuleType("knowledge_base.indexing.embeddings")
    fake_embeddings.get_embedder = lambda: _FakeEmbedder()
    monkeypatch.setitem(
        sys.modules, "knowledge_base.indexing.embeddings", fake_embeddings
    )

    import knowledge_base.indexing.milvus_client as milvus_module
    from knowledge_base.faq import FAQIndex

    monkeypatch.setattr(milvus_module, "get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(
        milvus_module, "ensure_collection_loaded", lambda client, name: None
    )

    def _fake_insert(rows: list) -> int:
        calls.append(("insert", len(rows)))
        return len(rows)

    monkeypatch.setattr(milvus_module, "insert_faq_rows", _fake_insert)

    index = FAQIndex()
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


# ==================== FAQ 抽取残留过滤（scripts/build_faq.py）====================


def test_build_faq_strips_trailing_section_title() -> None:
    """答案的抽取残留必须清掉：尾部章节标题 / 控件文字、**开头多余的问句**、
    以及被页面渲染断开的拉丁字母 / 数字 token。

    抽取按编号标记切块，页面上的小标题（"其他问题""邮件客户端问题"）与
    "查看原图"页的按钮文字（"返回 原图 /"）不属于任何条目，
    会黏在**上一条答案末尾**；并列问法或下一问的原文会挤进**答案开头**；
    渲染还会把 token 断开（"i map.exmail.qq.com"、"E xchange"、"5 0 MB"）。
    而 FAQ 命中后答案**原样直返用户**，故这些噪声代价高。
    本用例的样本取自改造前 `data/raw/faq.json` 的真实残留（见 07 登记 G9 / I5）。
    """
    from scripts.build_faq import (
        _fix_broken_tokens,
        _strip_leading_question,
        _strip_trailing_residue,
    )

    # ① 尾部：章节标题 / 页面控件文字
    assert (
        _strip_trailing_residue("此时微信中有可能看不到提醒。 其他问题")
        == "此时微信中有可能看不到提醒。"
    )
    assert (
        _strip_trailing_residue("可以至电子论文平台查看。 返回 原图 /")
        == "可以至电子论文平台查看。"
    )

    # ② 开头：多出来的一问（问句字段本身是干净的，残留落在答案头部）
    assert (
        _strip_leading_question(
            "在哪里提交我的电子版学位学位论文？ 提交网址为 https://thesis.fudan.edu.cn/"
        )
        == "提交网址为 https://thesis.fudan.edu.cn/"
    )

    # ③ 渲染断开的 token：只合"单字母 + 空格 + 词"与"数字 + 空格 + 数字"，
    #    正常的英文短语（词长 > 1）不受影响
    assert (
        _fix_broken_tokens("最大为5 0 MB，发出后 2 4 小时可撤回；见 i map.exmail.qq.com")
        == "最大为50 MB，发出后 24 小时可撤回；见 imap.exmail.qq.com"
    )
    assert _fix_broken_tokens("the exam is over") == "the exam is over"


def test_build_faq_keeps_short_answer_intact() -> None:
    """剥离不得把一条短答案误删成空：剥完不足最小长度就保留原样。"""
    from scripts.build_faq import _strip_trailing_residue

    assert _strip_trailing_residue("可以") == "可以"
    assert _strip_trailing_residue("由研究生院和院系做出规定。") == (
        "由研究生院和院系做出规定。"
    )


# ==================== 附件元数据 sidecar ====================


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


# ==================== 大 PDF 解析可观测性 ====================


def test_parse_file_task_reports_elapsed(tmp_path: Path) -> None:
    """单文件解析耗时随结果返回：主进程靠它汇总"最慢文件"。"""
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


# ==================== 重依赖链可导入性（I1 回归） ====================


def test_torch_and_torchvision_are_importable() -> None:
    """torch 与 torchvision 必须能**一起**导入——这是一条依赖组合的回归断言。

    I1 的根因正是这一对不匹配：lock 里 torch 来自 cu126 通道，而 torchvision
    因为是间接依赖（unstructured[pdf] → unstructured-inference → timm），
    没吃到 [tool.uv.sources] 的 index 映射，从默认镜像装成了 PyPI 的 CPU 构建。
    Linux 上 `import torchvision` 直接抛
    `RuntimeError: operator torchvision::nms does not exist`，连带 transformers
    无法惰性导入 TrainingArguments（CI 首跑 2 项 FAQ 用例因此失败）。

    断言只卡"能导入"这一档：导入即崩才是真正的故障模式；
    本机 Windows 历史装的是 CPU 版 torchvision 也能正常工作，不必强求构建族一致。
    """
    import torch
    import torchvision

    assert torch.__version__
    assert torchvision.__version__
