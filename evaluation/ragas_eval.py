"""RAGAS 评测：检索质量（组件级）+ 端到端（整体）两个层次。

用法：
    python evaluation/ragas_eval.py --data data/eval/rag_eval.json

评测集格式（JSON 数组，UTF-8）：
    [
      {
        "id": "q001",
        "question": "本科生选课截止时间是什么时候？",
        "reference_answer": "第2教学周周日24:00前……",   # 人工标注的标准答案
        "user_profile": {"role": "本科生"}              # 可选，默认本科生
      }
    ]

裁判模型设计（对应架构文档模块四）：
- 跨模型族去偏：生成用 DeepSeek，裁判用百炼 Qwen——同族模型自评存在系统性偏好
  （倾向给自己/同门的输出高分），跨族组合消除该偏差
- 版本锁定（qwen3-max-2026-01-23 快照）：模型静默升级会让不同迭代的分数不可比，
  而纵向可复现性正是这套离线评测存在的意义（选型理由见 config/settings.py）
- temperature=0：评分确定性（同一回答多次评测结果必须一致）
- Embedding 本地化：Answer Relevancy 的向量化用本地 BGE-M3，
  该指标不涉及语义判断，无需跨族，也避免评测数据出境

RAGAS 0.4.3 的 API 迁移说明（v1 写法在此版本已不可用）：
`metrics=[faithfulness, ...]`（`ragas.metrics` 的模块级单例）在 0.4.3 会直接抛
`TypeError: All metrics must be initialised metric objects`。
改法有两条，本项目选后者：
① 退回 `evaluate()` + `ragas.metrics._*` 私有模块里的旧单例——能在 0.4.3 跑通，
   但依赖私有路径，官方 0.5 起随时可能删除；
② 用公开的 `ragas.metrics.collections` 新指标 + `ascore()` 逐条打分，自己聚合。
   （新指标继承自 `collections.base.BaseMetric`，与 `evaluate()` 校验用的 legacy
   `Metric` 不是同一套基类，所以**不能**混用：要么全走新 API，要么全走旧的。）
本模块实现的是 ②，指标定义、裁判与 embedding 与官方默认一致。

数据脱敏：所有发送给外部裁判 API 的文本先替换手机号 / 学号。
"""

import argparse
import asyncio
import inspect
import json
import math
import re

import pandas as pd
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import InstructorBaseRagasLLM, llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config.settings import settings
from infra.llm_clients import build_judge_client

# 并发上限：评测是 IO 密集型（等裁判 API 返回），并发能显著压缩总耗时；
# 但裁判端点有速率限制，8 路是实测不触发限流的上限
MAX_CONCURRENCY = 8

# 五项核心指标的目标值与不达标时的诊断方向
TARGETS: dict = {
    "context_precision": (0.80, "检索排序有问题 → 优化 Reranker 或调整混合检索权重"),
    "context_recall": (0.75, "关键信息没检索到 → 优化切分策略或扩大检索范围"),
    "faithfulness": (0.85, "模型在编造答案 → 加强 Prompt 约束或更换更强的生成模型"),
    "answer_relevancy": (0.85, "回答跑题或废话多 → 优化 Prompt 中的输出格式约束"),
    "answer_correctness": (0.80, "端到端答案不准确 → 综合排查检索 + 生成两个环节"),
}


def sanitize_text(text: str) -> str:
    """脱敏：替换手机号、学号等个人敏感信息。

    评测数据可能来自真实学生提问，进入外部裁判 API 前必须清洗，
    降低个人信息泄露风险。
    """
    text = re.sub(r"1[3-9]\d{9}", "[手机号]", text)
    text = re.sub(r"\b\d{8,11}\b", "[学号]", text)
    return text


def sanitize_dataset(raw_data: dict) -> dict:
    """对整个评测数据集脱敏（retrieved_contexts 为二维列表需逐层处理）。"""
    sanitized: dict = {}
    for key, values in raw_data.items():
        if key == "retrieved_contexts":
            sanitized[key] = [[sanitize_text(ctx) for ctx in ctxs] for ctxs in values]
        elif key in ("response", "user_input", "reference"):
            sanitized[key] = [sanitize_text(value) for value in values]
        else:
            sanitized[key] = values
    return sanitized


def build_judge() -> InstructorBaseRagasLLM:
    """裁判模型：百炼 qwen3-max 快照版，temperature=0。

    客户端由基础设施层的 `build_judge_client()` 提供（缺密钥时它直接报错，
    不走降级链——评测拿不到裁判就该立刻失败）。裁判独立配置的选型理由见 settings.py。

    客户端**必须是异步的**：新指标的 ascore() 走 agenerate()，
    传入同步 OpenAI 客户端会直接抛 "Cannot use agenerate() with a synchronous
    client"。异步客户端也让下面的并发打分真正并行（同步客户端只能串行等待）。
    """
    return llm_factory(
        settings.JUDGE_MODEL,
        provider="openai",
        client=build_judge_client(),
        temperature=0,  # 裁判必须 0：同一输入多次评测结果需一致
        # 1024 会在 answer_correctness 上触顶（该指标要逐句比对答案与标准答案，
        # 输出结构比其余四项长得多），实测抛 IncompleteOutputException 导致整格缺失
        max_tokens=2048,
    )


def build_judge_embeddings() -> HuggingFaceEmbeddings:
    """评测用 embedding：本地 BGE-M3（answer relevancy 指标的向量化）。

    复用检索侧已下载的 models/bge-m3，而不是另拉一个 bge-large-zh-v1.5：
    ① 离线可复现——评测不该依赖"临时下载一个模型"，且本机 HF 直连不稳定；
    ② 口径一致——answer_relevancy 的向量空间与检索侧同源，避免两套语义空间混用。

    该目录同时含 sentence-transformers 配置（modules.json / 1_Pooling），
    故可被 RAGAS 内置的 HuggingFaceEmbeddings 直接加载，无需再经 LangChain 包装。
    """
    return HuggingFaceEmbeddings(model=settings.EMBED_MODEL_NAME)


def build_metrics(
    llm: InstructorBaseRagasLLM, embeddings: HuggingFaceEmbeddings
) -> list:
    """五项核心指标：构造时注入裁判与 embedding（RAGAS 0.4.3 的接口约定）。

    指标顺序与 TARGETS 的输出顺序无关，但保持一致便于人工比对报告。
    """
    return [
        Faithfulness(llm=llm),  # 忠实度（最重要：编造检测）
        AnswerRelevancy(llm=llm, embeddings=embeddings),  # 答案相关性
        ContextPrecision(llm=llm),  # 上下文精确率（检索排序质量）
        ContextRecall(llm=llm),  # 上下文召回率（检索覆盖度）
        AnswerCorrectness(llm=llm, embeddings=embeddings),  # 端到端正确性
    ]


def metric_inputs(metric, row: dict) -> dict:
    """按指标 ascore 的形参名从一行数据里取参。

    五项指标需要的列并不相同（Faithfulness 不需要 reference，
    AnswerCorrectness 不需要 retrieved_contexts），用形参名做映射，
    增删指标时无需改动这里的代码。
    """
    parameters = inspect.signature(metric.ascore).parameters
    missing = [name for name in parameters if name not in row]
    if missing:
        raise KeyError(f"指标 {metric.name} 需要数据集缺少的列: {missing}")
    return {name: row[name] for name in parameters}


async def _score_pair(
    row_index: int, metric, row: dict, semaphore: asyncio.Semaphore
) -> tuple:
    """对「单行 × 单指标」打分，返回 (行号, 指标名, 分值或 NaN)。

    单点失败（裁判超时 / 返回无法解析的评分）记为该单元格 NaN 并继续，
    不让一条坏样本作废整轮评测——与 ragas 的 raise_exceptions=False 语义一致。
    """
    async with semaphore:
        try:
            result = await metric.ascore(**metric_inputs(metric, row))
            return row_index, metric.name, float(result.value)
        except Exception as score_error:
            print(
                f"[ragas_eval] 第 {row_index} 条 {metric.name} 打分失败: "
                f"{type(score_error).__name__}: {str(score_error)[:100]}"
            )
            return row_index, metric.name, math.nan


async def _score_rows(rows: list, metrics: list) -> list:
    """并发打分并按行汇总：返回 [{"user_input", 指标名...}, ...]。

    逐条产出进度：100 条 × 5 指标的全量评测动辄几十分钟，
    没有进度输出就无法判断是"卡住了"还是"只是慢"（v2 实测教训）。
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        _score_pair(index, metric, row, semaphore)
        for index, row in enumerate(rows)
        for metric in metrics
    ]

    scored: list = []
    for completed, future in enumerate(asyncio.as_completed(tasks), start=1):
        scored.append(await future)
        if completed % 25 == 0 or completed == len(tasks):
            print(f"[ragas_eval] 打分进度 {completed}/{len(tasks)}")

    details = [dict(row) for row in rows]
    for row_index, metric_name, value in scored:
        details[row_index][metric_name] = value
    return details


def _aggregate(details: list, metrics: list) -> dict:
    """按指标求均值（自动跳过打分失败的 NaN 单元格）。"""
    scores: dict = {}
    for metric in metrics:
        values = [
            row[metric.name]
            for row in details
            if not math.isnan(row.get(metric.name, math.nan))
        ]
        scores[metric.name] = round(sum(values) / len(values), 4) if values else math.nan
    return scores


def run_eval(raw_eval_data: dict) -> dict:
    """执行 RAGAS 五项指标评测并输出结果对照目标值。

    :param raw_eval_data: 键为 user_input / response / retrieved_contexts / reference
        的字典（RAGAS 0.4.x 的列名；旧列名 question/answer/contexts/ground_truth
        会被官方工具隐式改名，此处显式使用新列名以免依赖隐式转换）
    :return: {"scores": {指标: 均值}, "details": [逐条明细]}
    """
    columns = sanitize_dataset(raw_eval_data)
    rows = [
        {key: columns[key][index] for key in columns}
        for index in range(len(columns["user_input"]))
    ]
    metrics = build_metrics(build_judge(), build_judge_embeddings())

    details = asyncio.run(_score_rows(rows, metrics))
    scores = _aggregate(details, metrics)

    print("\n===== RAG 评测结果 =====")
    for metric_name, score in scores.items():
        target, diagnosis = TARGETS.get(metric_name, ("-", ""))
        if math.isnan(score):
            print(f"  {metric_name:22s}: 全部样本打分失败")
            continue
        status = "PASS" if score >= target else "FAIL"
        print(
            f"  {metric_name:22s}: {score:.4f} (目标 > {target}) [{status}] {diagnosis}"
        )

    # 逐条明细导出：Bad Case 定位与回归比对的数据基础
    report_path = "rag_eval_report.csv"
    pd.DataFrame(details).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n逐条明细已保存至 {report_path}")
    return {"scores": scores, "details": details}


def run_end_to_end_eval(eval_data_path: str) -> None:
    """端到端评测：每条问题完整跑 Agent 流水线后再用 RAGAS 打分。

    流程：加载评测集 → 逐条 invoke（收集回答与检索上下文）
    → 汇总为 RAGAS 输入格式 → 打分。

    :param eval_data_path: 评测集 JSON 路径
    """
    from agent.graph import invoke

    with open(eval_data_path, encoding="utf-8") as file:
        eval_qa_pairs = json.load(file)

    raw_data: dict = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for qa in eval_qa_pairs:
        result = invoke(
            user_input=qa["question"],
            user_id="eval_user",
            session_id=f"eval_{qa.get('id', id(qa))}",  # 独立会话避免互相污染记忆
            user_profile=qa.get("user_profile", {"role": "本科生"}),
        )
        # 检索上下文从 RetrievedChunk 字典列表中提取纯文本
        contexts = [chunk["text"] for chunk in result.get("retrieved_contexts", [])]

        raw_data["user_input"].append(qa["question"])
        raw_data["response"].append(result["final_response"])
        raw_data["retrieved_contexts"].append(contexts)
        raw_data["reference"].append(qa["reference_answer"])
        print(
            f"[ragas_eval] 完成 {len(raw_data['user_input'])}/{len(eval_qa_pairs)}: {qa['question'][:30]}"
        )

    run_eval(raw_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小旦答 RAGAS 端到端评测")
    parser.add_argument(
        "--data",
        default="data/eval/rag_eval.json",
        help="评测集路径（JSON 数组，含 id/question/reference_answer/user_profile）",
    )
    args = parser.parse_args()

    run_end_to_end_eval(args.data)
