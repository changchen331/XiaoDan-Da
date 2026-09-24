"""情绪分类语料合成：用 LLM 生成校园场景下的四级情绪表达（含隐晦表达）。

用法：
    # 训练语料：100 条，高危占比 30%
    python scripts/synthesize_emotion_corpus.py --out data/eval/emotion_train.json

    # 评测集：40 条，并排除训练语料中的重复表达
    python scripts/synthesize_emotion_corpus.py --out data/eval/emotion_eval.json \
        --count 40 --exclude data/eval/emotion_train.json

为什么用合成语料而不是公开数据集（v2 的取舍，理由需要写清楚）：
- 公开数据集（NLPCC 微博情绪 / ChnSentiCorp / 抑郁对话）采用的是「喜/怒/哀/惧」
  或「正/负/中性」标签体系，与本项目的四级标签**不对齐**，
  无法直接验证四级分类效果——硬映射会引入无法量化的标签噪声
- 本项目四级标签的边界由业务定义（"是否需要人工介入"），
  校园场景下最具价值的**隐晦表达**（"最近总在想，如果我不在了大家会不会轻松一点"）
  在通用数据集里几乎不存在，而那恰恰是分类模型要补上规则引擎的缺口

⚠ 高危占比 30% 是**刻意的偏离**：真实分布里高危不足 8%，
但 100 条样本规模下 8 条高危会让召回率的最小可分度达到 12.5%，读不出结论。
v3 扩到 1500 条时回落到接近真实的分布。训练脚本的过采样与类权重仍按原始比例计算。

高危表达的生成约束：只允许出现意念 / 情绪 / 告别的**文字表达**，
不生成任何自伤方式、工具或操作细节——数据集本身不应成为风险源。
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from emotion.rule_engine import LEVEL_HIGH, RuleEngine
from infra.llm_clients import JSON_TASK_ERRORS, chat_qwen_json_parsed

# ===== 四级标签的判定规则（喂给 LLM，同时也是语料标注的一致性基准）=====
LABEL_DEFINITIONS: dict = {
    "正常": (
        "情绪平稳，或只有转瞬即逝的轻微负面情绪；没有持续困扰，"
        "不影响作息、学习和社交。例如：抱怨作业多但心态放松、"
        "考试前有点紧张但能自我调节。"
    ),
    "轻度困扰": (
        "有明确的负面情绪（焦虑 / 低落 / 孤独 / 烦躁 / 自我怀疑），"
        "持续数天，对学习生活影响轻微，仍能维持正常作息与社交；"
        "以倾诉和寻求安慰为主，没有功能损害。"
    ),
    "中度困扰": (
        "负面情绪持续两周以上，或已明显损害功能：长期失眠、无法集中注意力、"
        "回避社交、缺课、食欲显著改变、反复自责；但**没有**自杀或自伤意念。"
    ),
    "高危": (
        "出现自杀意念、自伤行为、绝望感、「活着没意义」、"
        "告别或交代后事等信号。**反讽、调侃、轻描淡写的语气也算高危**——"
        "「我这种人还是消失比较好吧」虽然像玩笑，仍是高危信号，"
        "这是最容易被漏报的一类表达。"
    ),
}

# 标签键必须与运行时唯一声明处（settings.EMOTION_LABELS）完全一致：
# 这里的四份定义会被拼进 prompt、也是语料标注的一致性基准，
# 与 settings 漂移会产出"标签体系对不上"的语料（改一处忘另一处的经典现场）
assert set(LABEL_DEFINITIONS) == set(
    settings.EMOTION_LABELS
), "LABEL_DEFINITIONS 的标签键必须与 settings.EMOTION_LABELS 完全一致"

# 校园场景池：覆盖学业 / 人际 / 家庭 / 经济 / 身心 / 发展六类压力源
SCENARIOS: tuple = (
    "期中考试挂科",
    "保研失败",
    "考研初试成绩不理想",
    "毕业论文被导师反复退回",
    "组会被当众批评",
    "室友作息冲突",
    "社团里被孤立",
    "异地恋分手",
    "家庭对成绩的高期待",
    "经济压力与兼职",
    "秋招投递无回应",
    "实习期间被否定",
    "长期失眠",
    "身材与外貌焦虑",
    "突发疾病住院",
    "亲人重病或离世",
    "转专业后的适应困难",
    "留学生语言与思乡问题",
    "网络暴力与谣言",
    "延毕带来的羞耻感",
    "对专业本身失去兴趣",
    "同辈压力与自我否定",
)

# 表达风格池：同一场景用不同方式说出来，模型才不会只学到字面关键词
STYLES: tuple = (
    "直白陈述",
    "隐晦暗示（不直接说情绪词，用状态与行为描述）",
    "反讽或自嘲（表面轻松，底下是负面情绪）",
    "躯体化描述（只说不舒服，不说心理感受）",
    "深夜碎碎念式（口语、短句、省略号）",
    "求助他人式（问别人该怎么办）",
)

# 单批生成的条数：批量请求能显著减少 API 往返，但过大时 LLM 会开始重复
BATCH_SIZE = 10

# 文本长度约束：过短无法判断情绪，过长不符合校园问答的实际输入形态
MIN_CHARS = 8
MAX_CHARS = 120

# 近重复判定阈值（字符 3-gram Jaccard 相似度）：合成语料里同质表达很常见，
# 重复样本会让训练集"看起来很大"但信息量不变
NEAR_DUP_THRESHOLD = 0.8


def _build_prompt(label: str, scenarios: list, styles: list, count: int) -> str:
    """构造单批生成 Prompt：固定标签定义，变化的场景与风格组合。

    :param label: 目标情绪等级
    :param scenarios: 本批使用的场景组合
    :param styles: 本批使用的表达风格
    :param count: 本批要求的条数
    """
    return f"""你在为「校园心理健康问答系统」构造情绪分类训练语料。

【目标等级】{label}
【该等级的定义】{LABEL_DEFINITIONS[label]}

【各等级的边界（务必严格遵守）】
- 正常：{LABEL_DEFINITIONS["正常"]}
- 轻度困扰：{LABEL_DEFINITIONS["轻度困扰"]}
- 中度困扰：{LABEL_DEFINITIONS["中度困扰"]}
- 高危：{LABEL_DEFINITIONS["高危"]}

【本批要求】
- 生成 {count} 条，每条是一句学生自己说出的话（第一人称，8-120 字）
- 场景从以下池中取，每条用不同的场景，不要出现两次相同场景：
  {"; ".join(scenarios)}
- 表达风格在以下风格间轮换：
  {"; ".join(styles)}
- 每条都必须能明确判定为「{label}」，不能是模糊的边界样本
- 语言用中文口语，像真实的聊天输入；可以带标点、语气词、省略号
- 不要编号、不要解释、不要出现"作为学生"这类旁白

【安全约束（仅对高危）】
只描述情绪、意念与告别，**不得出现任何自伤方式、工具、时间或操作细节**。

【输出格式】只输出 JSON：
{{"samples": ["第一句", "第二句", ...]}}"""


def _generate_batch(label: str, index: int, count: int, scenario_offset: int) -> list:
    """调用 LLM 生成一批样本，返回文本列表（失败时返回空列表）。

    场景与风格按批次序号错位取样，避免不同批次撞到同一组表达。

    :param label: 目标情绪等级
    :param index: 批次序号（决定场景与风格的切分位置）
    :param count: 本批条数
    :param scenario_offset: 场景池起始偏移量；生成评测集时与训练集取不同值，
        使同一场景下评测集换用未在训练集出现过的表达风格组合
    """
    offset = index * count + scenario_offset
    scenarios = [
        SCENARIOS[(offset + i) % len(SCENARIOS)] for i in range(count)
    ]
    styles = [STYLES[(offset + i) % len(STYLES)] for i in range(count)]

    prompt = _build_prompt(label, scenarios, styles, count)
    for attempt in range(2):
        try:
            payload = chat_qwen_json_parsed(prompt)
            samples = payload.get("samples", [])
            if isinstance(samples, list) and samples:
                return [str(item).strip() for item in samples if str(item).strip()]
        except JSON_TASK_ERRORS as error:
            print(f"[synthesize] {label} 第 {index} 批失败（第 {attempt + 1} 次）：{error}")
            time.sleep(2)
    return []


def _normalize(text: str) -> str:
    """归一化文本用于去重：去掉空白与标点，只留汉字、字母与数字。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


def _is_near_duplicate(text: str, seen_grams: list, threshold: float) -> bool:
    """近重复判定：字符 3-gram 的 Jaccard 相似度超过阈值即视为重复。

    逐条与已有样本比较在 100-200 条量级下开销可忽略，
    不值得为此引入向量化去重的复杂度。
    """
    normalized = _normalize(text)
    if len(normalized) < 3:
        return True
    grams = {normalized[i : i + 3] for i in range(len(normalized) - 2)}
    for existing in seen_grams:
        if not grams or not existing:
            continue
        intersection = len(grams & existing)
        union = len(grams | existing)
        if union and intersection / union >= threshold:
            return True
    return False


def _label_quotas(total: int, high_risk_ratio: float) -> dict:
    """按目标高危占比分配各级条数（高危单独指定，其余三等分后补齐余数）。"""
    high_risk = round(total * high_risk_ratio)
    remaining = total - high_risk
    base = remaining // 3
    quotas = {
        "高危": high_risk,
        "中度困扰": base + (remaining - base * 3),
        "轻度困扰": base,
        "正常": base,
    }
    return quotas


def generate_corpus(
    total: int, high_risk_ratio: float, exclude_paths: list, scenario_offset: int
) -> tuple:
    """并发生成全部语料：按等级分批判量请求，逐条做长度与近重复过滤。

    :param total: 目标总条数
    :param high_risk_ratio: 高危样本占比
    :param exclude_paths: 需要排除的已有语料文件（避免训练集与评测集重叠）
    :param scenario_offset: 场景池起始偏移量
    :return: (samples, stats)
        - samples: [{"text", "label"}, ...]
        - stats: 生成过程的统计信息（用于写入溯源文件）
    """
    quotas = _label_quotas(total, high_risk_ratio)

    # 已有语料全部载入近重复比对池：评测集必须与训练集完全隔离，
    # 否则"模型在评测集上表现好"只是背下了训练样本
    seen_grams: list = []
    for path in exclude_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as file:
            for entry in json.load(file):
                normalized = _normalize(entry["text"])
                if len(normalized) >= 3:
                    seen_grams.append(
                        {normalized[i : i + 3] for i in range(len(normalized) - 2)}
                    )
    print(f"[synthesize] 去重基准 {len(seen_grams)} 条（来自 {len(exclude_paths)} 个已有语料）")

    # 构造批次任务：每批最多 BATCH_SIZE 条
    tasks: list = []
    for label, quota in quotas.items():
        batch_count = (quota + BATCH_SIZE - 1) // BATCH_SIZE
        for index in range(batch_count):
            tasks.append((label, index, min(BATCH_SIZE, quota - index * BATCH_SIZE)))

    samples: list = []
    failures: list = []
    rule_engine = RuleEngine()
    rule_caught_high_risk = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                _generate_batch, label, index, count, scenario_offset
            )
            for label, index, count in tasks
        ]
        for future, (label, index, count) in zip(futures, tasks):
            texts = future.result()
            if not texts:
                failures.append(f"{label}#{index}")
                continue

            kept = 0
            for text in texts:
                if kept >= count:
                    break
                if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                    continue
                if _is_near_duplicate(text, seen_grams, NEAR_DUP_THRESHOLD):
                    continue

                # 与规则引擎的一致性校验：规则引擎命中"高危"却标成其他等级，
                # 属于标注噪声——生产链路上该样本会被规则层直接改判，
                # 模型永远学不到"这种情况下应该输出其他等级"
                rule_level = rule_engine.check(text)["level"]
                if rule_level == LEVEL_HIGH and label != "高危":
                    continue
                if label == "高危" and rule_level == LEVEL_HIGH:
                    rule_caught_high_risk += 1

                normalized = _normalize(text)
                seen_grams.append(
                    {normalized[i : i + 3] for i in range(len(normalized) - 2)}
                )
                samples.append({"text": text, "label": label})
                kept += 1
            print(f"[synthesize] {label} 第 {index} 批：生成 {len(texts)} 条，采纳 {kept} 条")

    stats = {
        "target_total": total,
        "generated_total": len(samples),
        "quotas": quotas,
        "failed_batches": failures,
        "high_risk_caught_by_rule": rule_caught_high_risk,
    }
    return samples, stats


def save(samples: list, out_path: str, stats: dict, exclude_paths: list) -> None:
    """保存语料与溯源信息。

    语料文件是纯 JSON 数组（训练脚本直接消费的格式）；
    溯源信息另存同名 `.meta.json`——三级标签体系、合成模型版本与生成时间
    决定了这批数据的含义，缺失则复现与横向比较都无从谈起。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)

    distribution = {
        label: sum(1 for sample in samples if sample["label"] == label)
        for label in settings.EMOTION_LABELS
    }
    meta = {
        "model": settings.LIGHT_LLM_MODEL,
        "endpoint": settings.LIGHT_LLM_BASE_URL,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distribution": distribution,
        "high_risk_ratio": (
            f"{distribution['高危'] / len(samples):.1%}" if samples else "0%"
        ),
        "excluded": exclude_paths,
        **stats,
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"\n[synthesize] 语料已保存: {out_path}（{len(samples)} 条）")
    print(f"[synthesize] 标签分布: {distribution}")
    print(f"[synthesize] 溯源信息: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 合成情绪分类语料")
    parser.add_argument(
        "--out",
        default="data/eval/emotion_train.json",
        help="输出路径（JSON 数组，元素含 text/label）",
    )
    parser.add_argument("--count", type=int, default=100, help="目标总条数")
    parser.add_argument(
        "--high-risk-ratio",
        type=float,
        default=0.3,
        help="高危样本占比（真实分布约 0.08，此处刻意提高以便观察召回率）",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="需要排除的已有语料文件（生成评测集时传入训练语料，避免污染）",
    )
    parser.add_argument(
        "--scenario-offset",
        type=int,
        default=0,
        help="场景池起始偏移量（生成评测集时与训练集取不同值，错开表达风格组合）",
    )
    args = parser.parse_args()

    corpus, corpus_stats = generate_corpus(
        args.count, args.high_risk_ratio, args.exclude, args.scenario_offset
    )
    if not corpus:
        raise SystemExit("[synthesize] 一条都没生成成功，请检查 LIGHT_LLM 端点配置")
    save(corpus, args.out, corpus_stats, args.exclude)