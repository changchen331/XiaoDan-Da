"""切分策略对照实验：把"分类型切分更好"从一句说法变成一组数字。

用法：
    python evaluation/chunk_experiment.py --data data/eval/rag_eval.json

被验证的说法：chunker.py 此前写着"实测 Recall@10 提升 12%"，
但仓库里没有任何实验记录支撑这个数字——它属于 v1 自查清单里的"未验证项"。
本脚本用评测集把它变成可复现的实测值，无论结论是证实还是推翻。

三个对照策略（同一份解析文本，只改切分边界）：
1. 固定长度硬切 768 / 重叠 76 —— 朴素基线，完全不看文档结构
2. 统一递归切分 —— 保留段落与句子边界，但所有文档走同一套参数（不看文档类型）
3. 分类型切分（线上策略）—— FAQ 按问答对、通知按标题、手册递归、表格整表

为什么只用 dense 单路 top-k 排序：
线上链路是 hybrid(RRF 融合) + Reranker，但本实验要隔离的**只有"切分"这一个变量**，
混入重排后两者的贡献无法分离。检索阶段召回不到的片段，重排也变不出来，
因此"该切分让证据更难被召回"这个结论的方向不受影响。

指标定义（Recall@k = top-k 中至少命中一个 gold chunk 的问题占比）：
- gold chunk = 正文中**包含该条 evidence 逐字片段**的 chunk（空白不敏感定位）
- 证据被切分边界切断时（两半分别落在相邻 chunk，没有任何一个 chunk 含完整证据），
  该条计为**未命中**而不是从分母里剔除——这是固定长度切分最典型的失效模式，
  剔除等于把它的代价藏起来。不可定位的条数单独统计，以便区分"被切碎"与"没排上来"
- 三条策略共用同一个分母（评测集全量条数），横向可比

检索池是**全部文档的全部 chunk**，不做元数据过滤，比线上条件更严苛：
过滤会缩小搜索空间、抬高召回率，去掉它得到的结论只会更保守。
"""

import argparse
import json
import os
import time

import numpy as np

from config.settings import settings
from knowledge_base.indexing.embeddings import get_embedder
from knowledge_base.preprocessing.chunker import chunk_by_type, chunk_fixed_length
from knowledge_base.preprocessing.corpus import iter_document_records, locate_span

# 对照策略：名称 → (切分函数, 说明)
STRATEGIES: dict = {
    "fixed_length": "固定长度硬切 768/76（朴素基线）",
    "uniform_recursive": "统一递归切分 768/76（结构感知、类型无关）",
    "by_type": "分类型切分（线上策略）",
}

# 编码批大小：BGE-M3 单批 8 条内部分片，这里按 400 条一批便于打印进度
ENCODE_BATCH = 400


def build_strategy_chunks(records: list) -> dict:
    """对同一批文档跑三种切分策略，返回各策略的 chunk 列表。

    :return: {策略名: [{"doc", "index", "text"}, ...]}
    """
    result: dict = {name: [] for name in STRATEGIES}

    for record in records:
        metadata = record["metadata"]
        doc_type = record["doc_type"]
        text = record["text"]

        pieces = {
            "fixed_length": chunk_fixed_length(text),
            # 统一递归切分复用"手册"分支：它正是类型无关的那套分隔符策略
            "uniform_recursive": chunk_by_type(text, "手册", metadata),
            "by_type": chunk_by_type(text, doc_type, metadata),
        }
        for name, chunks in pieces.items():
            for index, chunk in enumerate(chunks):
                result[name].append(
                    {"doc": record["path"], "index": index, "text": chunk["text"]}
                )

    return result


def locate_gold_indices(chunks: list, doc: str, evidence: str) -> list:
    """找出某文档中所有包含指定 evidence 的 chunk 在全局列表里的下标。

    一个 evidence 可能同时落在多个 chunk 里（重叠区）——它们都是合法的
    gold chunk，命中任意一个即算召回成功。
    """
    return [
        position
        for position, chunk in enumerate(chunks)
        if chunk["doc"] == doc and locate_span(chunk["text"], evidence) is not None
    ]


def encode_texts(texts: list) -> np.ndarray:
    """分批编码为归一化稠密向量矩阵（余弦相似度的前提是模长归一）。"""
    embedder = get_embedder()
    batches: list = []
    for start in range(0, len(texts), ENCODE_BATCH):
        batch = texts[start : start + ENCODE_BATCH]
        vectors = [item["dense"] for item in embedder.encode(batch)]
        batches.append(np.asarray(vectors, dtype=np.float32))
        print(f"[chunk_experiment] 编码 {min(start + ENCODE_BATCH, len(texts))}/{len(texts)}")

    matrix = np.vstack(batches)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 全零向量（空文本）不参与排序，避免除零
    return matrix / norms


def top_k_indices(scores: np.ndarray, top_k: int) -> list:
    """取相似度最高的 top_k 个下标（按分数降序）。"""
    if len(scores) <= top_k:
        return list(np.argsort(-scores))
    candidates = np.argpartition(-scores, top_k)[:top_k]
    return list(candidates[np.argsort(-scores[candidates])])


def evaluate_strategy(
    chunks: list, chunk_matrix: np.ndarray, query_matrix: np.ndarray, eval_items: list,
    top_k: int,
) -> dict:
    """对单一策略计算 Recall@1/@5/@k 与不可定位条数。"""
    # hits_at 的键即截断位置；top_k 一定在键中，保证表格最后一行有意义
    hits_at = {1: 0, 5: 0, top_k: 0}
    unlocatable = 0
    gold_counts: list = []

    for position, item in enumerate(eval_items):
        gold = locate_gold_indices(chunks, item["source"]["doc"], item["evidence"])
        gold_counts.append(len(gold))
        if not gold:
            # 证据被切分边界切断：没有任何 chunk 含完整证据，计为未命中
            unlocatable += 1
            continue

        ranked = top_k_indices(chunk_matrix @ query_matrix[position], top_k)
        for k in hits_at:
            if any(index in ranked[:k] for index in gold):
                hits_at[k] += 1

    total = len(eval_items)
    # 召回率以全量问题为分母：未定位到的条数已算作未命中
    return {
        "chunks": len(chunks),
        "avg_chars": round(float(np.mean([len(c["text"]) for c in chunks])), 1),
        "max_chars": max(len(c["text"]) for c in chunks),
        "recall@1": round(hits_at[1] / total, 4),
        "recall@5": round(hits_at[5] / total, 4),
        f"recall@{top_k}": round(hits_at[top_k] / total, 4),
        "unlocatable": unlocatable,
        "avg_gold_chunks": round(float(np.mean(gold_counts)), 2),
        "questions": total,
    }


def run(data_path: str, top_k: int, max_questions: int | None) -> dict:
    """实验主流程：加载评测集 → 三策略切分 → 编码 → 逐策略算指标。

    :param data_path: 评测集路径（由 scripts/synthesize_rag_eval.py 生成）
    :param top_k: 召回截断位置
    :param max_questions: 只跑前 N 条（冒烟用）
    """
    with open(data_path, encoding="utf-8") as file:
        eval_items = json.load(file)
    if max_questions:
        eval_items = eval_items[:max_questions]
    print(f"[chunk_experiment] 评测集: {len(eval_items)} 条，截断位置 top-{top_k}")

    records, failures = iter_document_records()
    if failures:
        print(f"[chunk_experiment] 注意：{len(failures)} 篇文档解析失败，其 chunk 不在检索池中")
    chunks_by_strategy = build_strategy_chunks(records)

    # 完整性校验：评测集里记录的 chunk_id 必须在 by_type 结果中确实含该证据，
    # 否则说明"评测集与索引同源"的前提被破坏（评测集是旧切分下生成的），
    # 本实验的所有结论都会失去意义——这种情况必须显式报错而不是继续算
    drift = 0
    for item in eval_items:
        source = item["source"]
        matched = [
            chunk
            for chunk in chunks_by_strategy["by_type"]
            if chunk["doc"] == source["doc"] and chunk["index"] == source["chunk_index"]
        ]
        if not matched or locate_span(matched[0]["text"], item["evidence"]) is None:
            drift += 1
    if drift:
        raise SystemExit(
            f"[chunk_experiment] {drift}/{len(eval_items)} 条评测样本的来源 chunk 对不上"
            f"——评测集与当前切分已不同源，请先用 synthesize_rag_eval.py 重新生成"
        )
    print("[chunk_experiment] 校验通过：评测集来源 chunk 与线上切分逐字一致")

    query_matrix = encode_texts([item["question"] for item in eval_items])

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_set": data_path,
        "questions": len(eval_items),
        "top_k": top_k,
        "embedding_model": settings.EMBED_MODEL_NAME,
        "strategies": {},
    }

    for name in STRATEGIES:
        chunks = chunks_by_strategy[name]
        print(f"\n[chunk_experiment] 策略 {name}：{len(chunks)} 个 chunk，开始编码")
        matrix = encode_texts([chunk["text"] for chunk in chunks])
        metrics = evaluate_strategy(chunks, matrix, query_matrix, eval_items, top_k)
        report["strategies"][name] = {**metrics, "description": STRATEGIES[name]}
        del matrix  # 及时释放：三个策略的矩阵同时驻留会白占几百 MB

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    """打印对照表，并给出"分类型切分 vs 朴素基线"的相对提升。"""
    top_k = report["top_k"]
    header = f"{'策略':<34}{'chunk数':>8}{'均长':>7}{'R@1':>8}{'R@5':>8}{'R@' + str(top_k):>8}{'未定位':>7}"
    print("\n===== 切分策略对照（dense 单路召回）=====")
    print(header)
    for name, metrics in report["strategies"].items():
        row = (
            f"{metrics['description']:<34}{metrics['chunks']:>8}"
            f"{metrics['avg_chars']:>7}{metrics['recall@1']:>8.4f}"
            f"{metrics['recall@5']:>8.4f}{metrics[f'recall@{top_k}']:>8.4f}"
            f"{metrics['unlocatable']:>7}"
        )
        print(row)

    baseline = report["strategies"]["fixed_length"][f"recall@{top_k}"]
    online = report["strategies"]["by_type"][f"recall@{top_k}"]
    if baseline > 0:
        lift = (online - baseline) / baseline
        report["comparison"] = {
            "baseline": "fixed_length",
            "target": "by_type",
            "absolute": round(online - baseline, 4),
            "relative": round(lift, 4),
        }
        print(
            f"\n分类型切分 vs 固定长度硬切：Recall@{top_k} "
            f"{baseline:.4f} → {online:.4f}（相对提升 {lift:.1%}，"
            f"绝对提升 {online - baseline:+.4f}）"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="切分策略对照实验")
    parser.add_argument(
        "--data", default="data/eval/rag_eval.json", help="RAG 评测集路径"
    )
    parser.add_argument("--top-k", type=int, default=10, help="召回截断位置")
    parser.add_argument(
        "--max-questions", type=int, default=None, help="只跑前 N 条（冒烟测试）"
    )
    parser.add_argument(
        "--out",
        default="data/eval/chunk_experiment.json",
        help="实验报告输出路径（供评测证据记录引用）",
    )
    args = parser.parse_args()

    experiment_report = run(args.data, args.top_k, args.max_questions)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(experiment_report, file, ensure_ascii=False, indent=2)
    print(f"\n[chunk_experiment] 报告已保存: {args.out}")