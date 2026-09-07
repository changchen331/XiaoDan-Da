"""知识库索引构建脚本：原始文档 → 解析 → 清洗 → 切分 → 向量化 → 入库。

用法：
    python scripts/build_index.py [数据目录]

数据目录约定（默认 data/raw/，目录名决定 doc_type 切分策略）：
    data/raw/
    ├── 手册/                  # 学生手册、培养方案等长文档（PDF/DOCX/TXT/MD）
    │   └── 学生手册.pdf
    ├── 通知/                  # 官网通知（爬虫产出或手动收集，TXT/HTML）
    │   └── <url_hash>.txt     #   爬虫产出自带【标题】【发布日期】头部
    ├── 表格/                  # 校历、时间表等表格类文档（PDF/HTML）
    │   └── 校历.pdf
    └── faq.json               # FAQ 标准问答库（结构见下）

faq.json 格式（数组，每项一个问答条目）：
    [
      {
        "question": "校园卡丢了怎么办？",
        "answer": "请携带有效证件到一卡通中心挂失补办……",
        "similar_questions": ["校园卡挂失流程", "饭卡丢了去哪里补"],
        "tags": ["校园卡"]
      }
    ]

通知类 txt 的元数据头部格式（爬虫自动产出，手动收集时照此编写）：
    【标题】关于2026年春季学期选课安排的通知
    【发布日期】2026-01-15
    【原文链接】https://jwc.fudan.edu.cn/xx/xxxx.htm

每次知识库更新后重跑本脚本，随后运行 evaluation/ragas_eval.py
确认检索质量无退化。
"""
import json
import os
import re
import sys

from config.settings import settings, get_current_semester
from knowledge_base.faq import get_faq_index
from knowledge_base.indexing.embeddings import get_embedder
from knowledge_base.indexing.milvus_client import (
    create_faq_collection,
    create_kb_collection,
    insert_chunks,
)
from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.cleaner import clean_text
from knowledge_base.preprocessing.parser import elements_to_text, parse_document

# 支持的文档扩展名：白名单之外的内容不进入管线
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".html", ".htm", ".docx", ".txt", ".md")

# 通知类 txt 的元数据头部行模式：【键】值
NOTICE_META_RE = re.compile(r"^【(标题|发布日期|原文链接)】(.*)$")

# 目录名 → doc_type 映射；未命中的目录一律按"手册"策略处理
DOC_TYPE_BY_DIR: dict = {"手册": "手册", "通知": "通知", "表格": "表格"}


def build(data_dir: str) -> None:
    """索引构建主流程：建表 → FAQ 索引 → 通用知识库入库。

    :param data_dir: 原始数据根目录（默认 data/raw）
    """
    print(f"[build_index] 数据目录: {data_dir}")

    # 第一步：确保两个 Collection 存在（幂等操作）
    create_kb_collection()
    create_faq_collection()

    # 第二步：FAQ 专用索引（独立的结构化数据入口）
    faq_path = os.path.join(data_dir, "faq.json")
    if os.path.exists(faq_path):
        with open(faq_path, encoding="utf-8") as file:
            faq_entries = json.load(file)
        row_count = get_faq_index().build(faq_entries)
        print(f"[build_index] FAQ 索引已写入 {row_count} 行（含相似问法）")

    # 第三步：通用知识库（按 doc_type 分目录批量处理）
    total_chunks = 0
    for dir_name in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        sub_dir = os.path.join(data_dir, dir_name)
        if not os.path.isdir(sub_dir):
            continue

        doc_type = DOC_TYPE_BY_DIR.get(dir_name, "手册")
        doc_chunks: list = []

        for file_name in sorted(os.listdir(sub_dir)):
            file_path = os.path.join(sub_dir, file_name)
            if not os.path.isfile(file_path) or not file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            doc_chunks.extend(_process_file(file_path, doc_type, dir_name))

        if doc_chunks:
            vectors = get_embedder().encode([chunk["text"] for chunk in doc_chunks])
            insert_chunks(doc_chunks, vectors)
            total_chunks += len(doc_chunks)
            print(f"[build_index] {dir_name}/ 已入库 {len(doc_chunks)} 个 chunk")

    print(f"[build_index] 完成：通用知识库共写入 {total_chunks} 个 chunk")
    print(f"[build_index] 当前学期: {get_current_semester()}，"
          f"时效性过滤将匹配该学期与「长期有效」的文档")


def _process_file(file_path: str, doc_type: str, topic_tag: str) -> list:
    """处理单个文档文件：解析 → 清洗 → 元数据提取 → 分类型切分。

    :param file_path: 文档路径
    :param doc_type: 文档类型（由所在目录名决定）
    :param topic_tag: 主题标签（由顶层目录名决定，用于管理后台分类）
    :return: chunk 列表 [{"text", "metadata"}, ...]
    """
    file_name = os.path.basename(file_path)

    # 通知类 txt 先解析头部元数据（爬虫产出的【标题】【发布日期】等）
    header_meta: dict = {}
    body_text: str
    if doc_type == "通知" and file_name.endswith(".txt"):
        header_meta, body_text = _parse_notice_file(file_path)
    else:
        elements = parse_document(file_path)
        body_text = elements_to_text(elements)

    cleaned = clean_text(body_text)

    # 元数据默认值：手动上传文档默认全员可见、长期有效
    metadata: dict = {
        "source_url": header_meta.get("原文链接", "manual_upload"),
        "doc_type": doc_type,
        "topic_tag": topic_tag,
        "target_audience": header_meta.get("target_audience", "全部"),
        "valid_semester": "长期有效",
        "publish_date": header_meta.get("发布日期", ""),
        "language": "zh",
        "title": header_meta.get("标题", os.path.splitext(file_name)[0]),
    }

    return chunk_by_type(cleaned, doc_type, metadata)


def _parse_notice_file(file_path: str) -> tuple:
    """解析通知类 txt 的元数据头部，返回 (元数据, 正文)。

    头部行格式：【标题】xxx / 【发布日期】xxx / 【原文链接】xxx，
    首个空行之后的内容视为正文。
    """
    with open(file_path, encoding="utf-8") as file:
        lines = file.read().split("\n")

    meta: dict = {}
    body_start = 0
    for i, line in enumerate(lines):
        matched = NOTICE_META_RE.match(line.strip())
        if matched:
            meta[matched.group(1)] = matched.group(2).strip()
        elif line.strip() == "" and meta:
            body_start = i + 1   # 元数据头部后的首个空行是正文起点
            break

    body = "\n".join(lines[body_start:])
    return meta, body


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
