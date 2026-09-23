"""知识库索引构建脚本：原始文档 → 解析 → 清洗 → 切分 → 向量化 → 入库。

用法：
    python scripts/build_index.py [数据目录] [--rebuild]

幂等语义（重跑安全）：
- 默认：通用知识库按 metadata["source_file"]（来源相对路径）**先删旧 chunk
  再插新内容**，FAQ 索引整体清空后覆盖重建——文档更新后直接重跑即可，
  不会累积重复数据
- --rebuild：先删除两个 Collection 再从头建。切分策略 / 向量维度 / schema
  变更时必须使用；既有索引若建立于 source_file 字段引入之前（老 chunk
  无法按来源定位），也需先跑一次 --rebuild 完成一次干净重建

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
    【适用对象】全部
    【生效学期】2026-2027秋季
其中【适用对象】【生效学期】是检索侧人群过滤与时效过滤的唯一数据来源，
缺失时按"全部 / 长期有效"处理（即不做任何过滤）。

每次知识库更新后重跑本脚本，随后运行 evaluation/ragas_eval.py
确认检索质量无退化。
"""

import argparse
import json
import os

from config.settings import get_current_semester
from knowledge_base.faq import get_faq_index
from knowledge_base.indexing.embeddings import get_embedder
from knowledge_base.indexing.milvus_client import (
    create_faq_collection,
    create_kb_collection,
    delete_chunks_by_source,
    insert_chunks,
    reset_collections,
)
from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.corpus import iter_document_records


def build(data_dir: str, rebuild: bool = False) -> None:
    """索引构建主流程：建表 → FAQ 索引 → 通用知识库入库。

    :param data_dir: 原始数据根目录（默认 data/raw）
    :param rebuild: 整库重建——先删除两个 Collection；默认为按来源幂等替换
    """
    print(f"[build_index] 数据目录: {data_dir}")

    if rebuild:
        print("[build_index] --rebuild：删除既有集合，整库重建")
        reset_collections()

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
    # 解析链路由 knowledge_base.preprocessing.corpus 统一提供：
    # 它同时被评测集合成脚本与切分对照实验复用，保证三处看到的 chunk 逐字一致
    # （评测集记录的"来源 chunk"必须能在索引里找到），且解析结果带磁盘缓存，
    # 重跑建库不必再花几十分钟重新解析 PDF
    records, failed_files = iter_document_records(data_dir)

    # 幂等：按来源先删旧 chunk 再插新内容，重跑不再累积重复数据。
    # --rebuild 已整体清空，无需逐篇删（也避免对空集合发无谓请求）
    if not rebuild:
        replaced = sum(
            delete_chunks_by_source(record["metadata"]["source_file"])
            for record in records
        )
        if replaced:
            print(f"[build_index] 幂等替换：已按来源清理 {replaced} 个旧 chunk")

    total_chunks = 0
    for dir_name in sorted({record["topic_tag"] for record in records}):
        doc_chunks: list = []
        for record in records:
            if record["topic_tag"] == dir_name:
                doc_chunks.extend(
                    chunk_by_type(
                        record["text"], record["doc_type"], record["metadata"]
                    )
                )
        if doc_chunks:
            vectors = get_embedder().encode([chunk["text"] for chunk in doc_chunks])
            insert_chunks(doc_chunks, vectors)
            total_chunks += len(doc_chunks)
            print(f"[build_index] {dir_name}/ 已入库 {len(doc_chunks)} 个 chunk")

    print(f"[build_index] 完成：通用知识库共写入 {total_chunks} 个 chunk")
    if failed_files:
        print(f"[build_index] 解析失败 {len(failed_files)} 个文件：")
        for file_name, reason in failed_files:
            print(f"    - {file_name}: {reason[:150]}")
    print(
        f"[build_index] 当前学期: {get_current_semester()}，"
        f"时效性过滤将匹配该学期与「长期有效」的文档"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建知识库索引（默认按来源幂等重建，--rebuild 整库重建）"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/raw",
        help="原始数据根目录（默认 data/raw）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="整库重建：先删除两个 Collection（schema / 向量维度变更时必须使用）",
    )
    arguments = parser.parse_args()
    build(arguments.data_dir, rebuild=arguments.rebuild)
