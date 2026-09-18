"""情绪检测评测：高危召回率是硬性上线指标（> 95% 不达标禁止上线）。

用法：
    python evaluation/emotion_eval.py --data data/eval/emotion_eval.json

评测集格式（JSON 数组，UTF-8）：
    [{"id": "e001", "text": "最近总是睡不着", "label": "中度困扰"}, ...]
标签取值：正常 / 轻度困扰 / 中度困扰 / 高危
（高危样本占比建议不低于 8%，重点覆盖隐晦表达与边界 case）

核心原则：宁可误报，不可漏报。
- 漏报（高危被判为正常/轻度）代价巨大：可能错过处于危机中的学生
- 误报（正常被判为高危）代价可控：多一次人工核实或一句多余关怀

报告重点：
1. 混淆矩阵最后一行的非对角线元素 = 被漏报的高危样本（最严重错误）
2. 整体准确率仅作参考：类别不平衡下会被"正常"类主导
"""

import argparse
import json

from sklearn.metrics import classification_report, confusion_matrix, recall_score

from config.settings import settings

# 四级标签（顺序与训练一致，同时作为混淆矩阵的行列顺序）
LABELS: list = list(settings.EMOTION_LABELS)

# 高危召回率目标：硬性上线门槛
HIGH_RISK_RECALL_TARGET = 0.95


def load_eval_data(path: str) -> list:
    """加载情绪评测集。

    :param path: JSON 数组路径，每项含 text 与 label
    :return: [{"text", "label"}, ...]
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run_emotion_eval(y_true: list, y_pred: list) -> dict:
    """执行情绪检测评测并输出三份报告。

    :param y_true: 真实标签列表
    :param y_pred: 预测标签列表
    :return: {"high_risk_recall", "passed"}，passed 为是否达到上线标准
    """
    # 1. 整体分类报告（逐类的 precision / recall / f1）
    print("===== 分类报告 =====")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    # 2. 高危召回率：唯一硬性上线指标
    high_risk_recall = (
        recall_score(y_true, y_pred, labels=["高危"], average=None, zero_division=0)[0]
        if "高危" in y_true
        else 0.0
    )
    passed = high_risk_recall >= HIGH_RISK_RECALL_TARGET
    print(
        f"高危召回率: {high_risk_recall:.2%} (目标 > {HIGH_RISK_RECALL_TARGET:.0%}) "
        f"[{'PASS' if passed else 'FAIL - 不能上线，必须继续优化'}]"
    )

    # 3. 混淆矩阵：行=真实，列=预测
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("===== 混淆矩阵（行=真实，列=预测，列序:", " / ".join(LABELS), "）=====")
    print(cm)

    # 4. 漏报高危明细统计：混淆矩阵最后一行的非对角线之和
    missed = int(
        cm[LABELS.index("高危")].sum() - cm[LABELS.index("高危")][LABELS.index("高危")]
    )
    print(f"\n被漏报的高危样本数: {missed}（最严重的错误类型）")

    return {"high_risk_recall": high_risk_recall, "passed": passed}


def main(eval_data_path: str) -> dict:
    """评测主流程：逐条执行两层情绪检测 → 汇总指标。

    :param eval_data_path: 评测集路径
    :return: run_emotion_eval 的返回值
    """
    from emotion.detector import detect_emotion

    eval_data = load_eval_data(eval_data_path)
    print(f"[emotion_eval] 加载评测集: {len(eval_data)} 条")

    y_true: list = []
    y_pred: list = []
    for entry in eval_data:
        detection = detect_emotion(entry["text"])
        y_true.append(entry["label"])
        y_pred.append(detection["level"])

    return run_emotion_eval(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小旦答情绪检测评测")
    parser.add_argument(
        "--data",
        default="data/eval/emotion_eval.json",
        help="评测集路径（JSON 数组，含 text/label 字段）",
    )
    args = parser.parse_args()

    main(args.data)
