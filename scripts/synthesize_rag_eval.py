"""RAG 评测集合成：从知识库 chunk 反向生成「学生真实会问的问题」。

用法：
    python scripts/synthesize_rag_eval.py --count 100 --out data/eval/rag_eval.json

为什么是合成而不是真实用户问题：
设计文档原本要求「200 条真实用户问题 + 200 条教务人工标注答案」，
但真实校园问答日志与教务标注资源本项目拿不到。改成"从 chunk 反向合成"后，
评测集反而多出两个真实问题集不具备的性质：
① 标准答案有据可查——每条答案都附带**逐字原文** evidence，
   ground_truth 不再依赖人工判断是否可信；
② **记录来源 chunk**——切分策略改变后，可按 evidence 在新切分里重新定位 chunk 边界，
   Recall@10 才谈得上精确计算；真实用户问题无法回溯到"本应被检索到的那一个 chunk"。

三条构造约束（不加约束的合成评测集会把结论带偏）：
1. 只从 target_audience=全部 且 valid_semester=长期有效 的文档取材：
   检索侧元数据过滤会直接排除其他文档，混进来会让"检索不到"变成过滤问题而非检索问题，
   而本评测集要测的是检索与切分本身；
2. 每篇文档最多取 --per-doc 个 chunk：培养方案类 PDF 一篇就贡献上千个 chunk，
   不限制会让 100 条问题全挤在少数几篇文档上，切分实验的结论失去代表性；
3. evidence 必须是原文逐字片段（空白不敏感定位），定位失败整条丢弃：
   宁可少几条，也不让"标准答案无据可查"的样本进入评测集。
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agent.llm_clients import LLMUnavailableError, chat_qwen_json, parse_json_response
from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.corpus import (
    iter_document_records,
    locate_span,
    normalize_for_match,
)

# 单条 chunk 的可用性门槛：过短的片段（标题行、页眉）支撑不了问答；
# 中文占比过低的片段多为表格数字或英文残留，向量语义稀薄
MIN_CHUNK_CHARS = 120
MIN_CJK_RATIO = 0.3

# 生成时的过采样系数：约六成候选会因"片段信息零碎"被模型跳过或被校验丢弃，
# 多取一些候选才能凑够目标条数
OVERSAMPLE_RATIO = 1.8

PROMPT_TEMPLATE = """以下是一段来自复旦大学官方文档的原文片段。

【文档标题】{title}
【文档类型】{doc_type}

请你扮演一名复旦大学在校学生，提出一个**只有依据这段原文才能回答**的问题。

要求：
1. question：学生真实会问的口语化问题，25 字以内；
   **问题必须自足**——不出现"这个""该""上述""本通知"等指代，
   读者只看问题本身就能知道在问什么；不要照抄原文措辞
2. reference_answer：标准答案，**只能使用片段中给出的信息**；
   片段只给出部分信息时，答案就只说这部分，不要补充任何片段之外的知识。
   答案要**完整自足**：包含关键的对象 / 时间 / 数字 / 条件，
   不要只回"合格""是的""可以"这类判断词——那样的标准答案无法用来判断
   生成答案是否准确（评测集里每条答案都会被拿去做 ground_truth）
3. evidence：从片段中**逐字摘录**一句支撑该答案的原文，必须与片段完全一致
   （含标点，不得改写、不得省略中间内容）
4. 若片段信息过于零碎、无法支撑一个有意义的问答（如只有目录、页码、无意义的表格数字），
   返回 {{"skip": true}}

【原文片段】
{chunk_text}

只输出 JSON：{{"question": "...", "reference_answer": "...", "evidence": "..."}}
或 {{"skip": true}}"""


def _cjk_ratio(text: str) -> float:
    """中文汉字占比（用于剔除表格数字类片段）。"""
    if not text:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", text)) / len(text)


def _is_usable_chunk(text: str) -> bool:
    """片段是否适合出题：长度与中文占比双门槛。"""
    return len(text) >= MIN_CHUNK_CHARS and _cjk_ratio(text) >= MIN_CJK_RATIO


def select_candidates(
    records: list, count: int, per_doc: int, seed: int
) -> list:
    """挑选出题用的候选 chunk：按文档过滤 + 每篇限流 + 确定性打乱。

    :param records: corpus.iter_document_records 的输出
    :param count: 目标条数（按 OVERSAMPLE_RATIO 放大候选量）
    :param per_doc: 每篇文档最多贡献几个 chunk
    :param seed: 随机种子（保证评测集可复现，而不是每次重新抽样）
    :return: [{"doc", "doc_type", "title", "chunk_index", "text"}, ...]
    """
    rng = random.Random(seed)
    candidates: list = []
    skipped_docs = 0

    for record in records:
        metadata = record["metadata"]
        # 约束 1：只取检索侧过滤条件一定放行的文档
        if metadata.get("target_audience", "全部") != "全部":
            skipped_docs += 1
            continue
        if metadata.get("valid_semester", "长期有效") != "长期有效":
            skipped_docs += 1
            continue

        chunks = chunk_by_type(record["text"], record["doc_type"], metadata)
        indexed = [
            (index, chunk["text"])
            for index, chunk in enumerate(chunks)
            if _is_usable_chunk(chunk["text"])
        ]
        rng.shuffle(indexed)  # 每篇文档内部随机取，避免总是取开头几个 chunk

        for chunk_index, text in indexed[:per_doc]:
            candidates.append(
                {
                    "doc": record["path"],
                    "doc_type": record["doc_type"],
                    "title": metadata.get("title", ""),
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

    rng.shuffle(candidates)
    print(
        f"[synthesize_rag] 候选 chunk {len(candidates)} 个"
        f"（来自跳过过滤的 {skipped_docs} 篇文档之外的全部合格文档）"
    )
    return candidates[: int(count * OVERSAMPLE_RATIO)]


def generate_one(candidate: dict) -> dict | None:
    """对单个 chunk 生成一条问答，返回 None 表示该候选不可用。

    校验点（任一不通过即丢弃）：
    - 模型主动 skip
    - 字段缺失或为空
    - evidence 无法在原文中定位（空白不敏感）——这说明摘录是改写或编造的，
      而 evidence 正是切分实验重新定位 gold chunk 的唯一依据，必须逐字可靠
    """
    prompt = PROMPT_TEMPLATE.format(
        title=candidate["title"],
        doc_type=candidate["doc_type"],
        chunk_text=candidate["text"],
    )
    for attempt in range(2):
        try:
            payload = parse_json_response(chat_qwen_json(prompt))
        except (LLMUnavailableError, ValueError) as error:
            print(f"[synthesize_rag] {candidate['doc']} 生成失败（第 {attempt + 1} 次）：{error}")
            time.sleep(2)
            continue

        if payload.get("skip"):
            return None
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("reference_answer", "")).strip()
        evidence = str(payload.get("evidence", "")).strip()
        if not question or not answer or not evidence:
            return None

        located = locate_span(candidate["text"], evidence)  # 必须能逐字定位
        if located is None:
            return None

        return {
            "question": question,
            "reference_answer": answer,
            "evidence": located,
            "chunk_id": f"{candidate['doc']}#{candidate['chunk_index']}",
        }
    return None


def build_eval_set(count: int, per_doc: int, seed: int) -> tuple:
    """主流程：选候选 → 并发生成 → 去重 → 组装评测集。

    :return: (eval_items, stats)
    """
    records, _ = iter_document_records()
    candidates = select_candidates(records, count, per_doc, seed)

    items: list = []
    used_questions: set = set()
    skipped = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(generate_one, candidates))

    for candidate, generated in zip(candidates, results):
        if generated is None:
            skipped += 1
            continue
        question_key = normalize_for_match(generated["question"])
        if question_key in used_questions:
            # 不同 chunk 可能问出同一句话（同一文档内的相似段落）：
            # 问题重复会让指标被同一信息需求重复计权
            skipped += 1
            continue
        used_questions.add(question_key)

        items.append(
            {
                "id": f"r{len(items) + 1:03d}",
                "question": generated["question"],
                "reference_answer": generated["reference_answer"],
                "user_profile": {"role": "本科生"},
                "source": {
                    "doc": candidate["doc"],
                    "title": candidate["title"],
                    "doc_type": candidate["doc_type"],
                    "chunk_index": candidate["chunk_index"],
                    "chunk_id": generated["chunk_id"],
                },
                "evidence": generated["evidence"],
            }
        )
        if len(items) >= count:
            break

    stats = {
        "requested": count,
        "generated": len(items),
        "candidates": len(candidates),
        "skipped": skipped,
        "per_doc_limit": per_doc,
        "seed": seed,
        "source_docs": len({item["source"]["doc"] for item in items}),
        "doc_type_distribution": _count_by_key(items, "doc_type"),
    }
    return items, stats


def _count_by_key(items: list, field: str) -> dict:
    """统计来源文档类型的分布（用于判断评测集是否被某一类文档主导）。"""
    distribution: dict = {}
    for item in items:
        key = item["source"][field]
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从知识库 chunk 合成 RAG 评测集")
    parser.add_argument(
        "--out", default="data/eval/rag_eval.json", help="输出路径（JSON 数组）"
    )
    parser.add_argument("--count", type=int, default=100, help="目标条数")
    parser.add_argument(
        "--per-doc", type=int, default=3, help="每篇文档最多贡献几个 chunk"
    )
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = parser.parse_args()

    eval_items, eval_stats = build_eval_set(args.count, args.per_doc, args.seed)
    if not eval_items:
        raise SystemExit("[synthesize_rag] 未能生成任何评测样本，请检查语料与 LLM 端点")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(eval_items, file, ensure_ascii=False, indent=2)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **eval_stats,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"\n[synthesize_rag] 评测集已保存: {args.out}（{len(eval_items)} 条）")
    print(f"[synthesize_rag] 覆盖文档 {eval_stats['source_docs']} 篇，"
          f"类型分布 {eval_stats['doc_type_distribution']}")
    print(f"[synthesize_rag] 溯源信息: {meta_path}")