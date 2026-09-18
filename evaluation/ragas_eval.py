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
- 跨模型族去偏：生成用 DeepSeek，裁判用 GPT-4o——同族模型自评
  存在系统性偏好（倾向给自己/同门的输出高分），跨族组合消除该偏差
- 版本锁定（gpt-4o-2024-08-06 快照）：不同迭代间的分数可复现可比较
- temperature=0：评分确定性（同一回答多次评测结果必须一致）
- Embedding 本地化：Answer Relevancy 的向量化用本地 BGE，
  该指标不涉及语义判断，无需跨族，也避免评测数据出境

数据脱敏：所有发送给外部裁判 API 的文本先替换手机号 / 学号。
"""

import argparse
import json
import re

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config.settings import settings

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
    """对整个评测数据集脱敏（contexts 为二维列表需逐层处理）。"""
    sanitized: dict = {}
    for key, values in raw_data.items():
        if key == "contexts":
            sanitized[key] = [[sanitize_text(ctx) for ctx in ctxs] for ctxs in values]
        elif key in ("answer", "question", "ground_truth"):
            sanitized[key] = [sanitize_text(value) for value in values]
        else:
            sanitized[key] = values
    return sanitized


def build_judge() -> LangchainLLMWrapper:
    """裁判模型：GPT-4o 快照版，temperature=0。"""
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.JUDGE_MODEL,
            temperature=0,
            max_tokens=1024,
        )
    )


def build_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """评测用 embedding：本地 BGE（answer relevancy 指标的向量化）。"""
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    )


def run_eval(raw_eval_data: dict) -> object:
    """执行 RAGAS 评测并输出结果对照目标值。

    :param raw_eval_data: 键为 question / answer / contexts / ground_truth 的字典
    :return: RAGAS EvaluationResult（to_pandas() 可得逐条明细）
    """
    eval_dataset = Dataset.from_dict(sanitize_dataset(raw_eval_data))

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,  # 忠实度（最重要：编造检测）
            answer_relevancy,  # 答案相关性
            context_precision,  # 上下文精确率（检索排序质量）
            context_recall,  # 上下文召回率（检索覆盖度）
            answer_correctness,  # 端到端正确性（与标准答案比对）
        ],
        llm=build_judge(),
        embeddings=build_judge_embeddings(),
    )

    print("\n===== RAG 评测结果 =====")
    for metric_name, score in results.items():
        target, diagnosis = TARGETS.get(metric_name, ("-", ""))
        status = "PASS" if score >= target else "FAIL"
        print(
            f"  {metric_name:22s}: {score:.4f} (目标 > {target}) [{status}] {diagnosis}"
        )

    # 逐条明细导出：Bad Case 定位与回归比对的数据基础
    report_path = "rag_eval_report.csv"
    results.to_pandas().to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n逐条明细已保存至 {report_path}")
    return results


def run_end_to_end_eval(eval_data_path: str) -> None:
    """端到端评测：每条问题完整跑 Agent 流水线后再用 RAGAS 打分。

    流程：加载评测集 → 逐条 invoke（收集回答与检索上下文）
    → 汇总为 RAGAS 输入格式 → 打分。

    :param eval_data_path: 评测集 JSON 路径
    """
    from agent.graph import invoke

    with open(eval_data_path, encoding="utf-8") as file:
        eval_qa_pairs = json.load(file)

    raw_data: dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for qa in eval_qa_pairs:
        result = invoke(
            user_input=qa["question"],
            user_id="eval_user",
            session_id=f"eval_{qa.get('id', id(qa))}",  # 独立会话避免互相污染记忆
            user_profile=qa.get("user_profile", {"role": "本科生"}),
        )
        # 检索上下文从 RetrievedChunk 字典列表中提取纯文本
        contexts = [chunk["text"] for chunk in result.get("retrieved_contexts", [])]

        raw_data["question"].append(qa["question"])
        raw_data["answer"].append(result["final_response"])
        raw_data["contexts"].append(contexts)
        raw_data["ground_truth"].append(qa["reference_answer"])
        print(
            f"[ragas_eval] 完成 {len(raw_data['question'])}/{len(eval_qa_pairs)}: {qa['question'][:30]}"
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
