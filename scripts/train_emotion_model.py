"""XLM-RoBERTa 情绪分类模型微调脚本（全量微调，单卡 4090 约 25-50 分钟）。

用法：
    python scripts/train_emotion_model.py [--train 路径] [--val 路径]

数据格式（JSON 数组，UTF-8）：
    [{"text": "最近考试压力好大", "label": "轻度困扰"}, ...]
标签取值：正常 / 轻度困扰 / 中度困扰 / 高危

数据构成（对应架构文档模块二）：
- 公开数据集：NLPCC 微博情绪 / ChnSentiCorp / 抑郁倾向对话（基础覆盖）
- 校园场景定制语料：心理学专业编写 + LLM 合成的隐晦表达（核心价值所在）

类别不平衡处理（高危样本天然只占约 8%，两级矫正）：
- 损失层：高危样本 3 倍 class weight，强迫模型为高危错误的梯度买单
- 数据层：高危样本 2 倍过采样，等比放大其在每个 epoch 中的出现频率

早停策略：监控验证集高危召回率（而非整体准确率——
整体准确率会被占多数的"正常"类主导，无法反映安全底线），
连续 2 个 epoch 无提升即停止，恢复最优权重。
"""
import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config.settings import settings

# 基座模型：跨语言预训练，同时覆盖中英文情绪表达（留学生场景）
BASE_MODEL_NAME = "xlm-roberta-base"
# 输入截断长度：情绪表达集中在文本前段
MAX_LENGTH = 128
# 各类别的损失权重：高危 ×3 应对类别不平衡
CLASS_WEIGHTS: dict = {"正常": 1.0, "轻度困扰": 1.0, "中度困扰": 1.5, "高危": 3.0}
# 高危样本过采样倍数
HIGH_RISK_OVERSAMPLE = 2


class EmotionDataset:
    """情绪分类数据集（PyTorch Dataset 协议实现）。"""

    def __init__(self, texts: list, labels: list, tokenizer) -> None:
        """初始化数据集。

        :param texts: 文本列表
        :param labels: 标签列表（已映射为类别 id 的整数）
        :param tokenizer: XLM-R 分词器
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """数据集规模。"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """取单条样本：分词 + 数值化，输出模型可直接消费的张量字典。"""
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


class WeightedTrainer(Trainer):
    """带类别权重的 Trainer：高危样本的损失按 3 倍计。

    原生 Trainer 使用等权 CrossEntropy，本子类只覆写 compute_loss 一处，
    其余训练逻辑（梯度累积 / 混合精度 / 日志）全部复用父类实现。
    """

    def __init__(self, *args, class_weights=None, **kwargs) -> None:
        """初始化并持有类别权重张量。

        :param class_weights: 形状 [num_labels] 的权重张量（已在正确设备上）
        """
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """加权损失计算：labels 从 inputs 中取出单独参与损失函数。"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits.view(-1, model.config.num_labels),
                             labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_corpus(path: str) -> tuple:
    """加载 JSON 语料并完成标签到 id 的映射。

    :param path: 语料文件路径
    :return: (texts, label_ids) 两个列表
    """
    label2id = {label: idx for idx, label in enumerate(settings.EMOTION_LABELS)}

    with open(path, encoding="utf-8") as file:
        entries = json.load(file)

    texts = [entry["text"] for entry in entries]
    label_ids = [label2id[entry["label"]] for entry in entries]
    return texts, label_ids


def oversample_high_risk(texts: list, labels: list) -> tuple:
    """高危样本 2 倍过采样（仅作用于训练集，验证集保持原始分布）。

    验证集必须反映真实分布，否则高危召回率指标会虚高。
    """
    high_risk_id = settings.EMOTION_LABELS.index("高危")
    pairs = list(zip(texts, labels))
    high_risk_pairs = [pair for pair in pairs if pair[1] == high_risk_id]

    oversampled = pairs + high_risk_pairs * (HIGH_RISK_OVERSAMPLE - 1)
    random.shuffle(oversampled)   # 打乱防止 batch 内类别聚集
    return [text for text, _ in oversampled], [label for _, label in oversampled]


def compute_metrics(eval_prediction) -> dict:
    """验证指标：整体准确率 + 高危召回率（早停监控项）。

    高危召回率 = 高危样本中被正确识别为高危的比例，
    是本任务的硬性上线指标（> 95%），漏报代价远高于误报。
    """
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids

    high_risk_id = settings.EMOTION_LABELS.index("高危")
    high_risk_recall = recall_score(
        labels, predictions, labels=[high_risk_id], average=None,
        zero_division=0,
    )[0]

    return {
        "accuracy": accuracy_score(labels, predictions),
        "high_risk_recall": high_risk_recall,
    }


def train(train_path: str, val_path: str | None) -> None:
    """微调主流程：加载语料 → 过采样 → 训练（加权损失 + 早停）→ 保存。

    :param train_path: 训练语料路径
    :param val_path: 验证语料路径；未提供时从训练集中分层切出 20%
    """
    import torch
    from sklearn.model_selection import train_test_split

    texts, labels = load_corpus(train_path)

    # 未指定独立验证集时，按类别分层切出 20%（保持各情绪类别比例）
    if val_path:
        val_texts, val_labels = load_corpus(val_path)
    else:
        texts, val_texts, labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels,
        )

    # 高危过采样（仅训练集）
    train_texts, train_labels = oversample_high_risk(texts, labels)
    label_distribution = {
        settings.EMOTION_LABELS[label_id]: train_labels.count(label_id)
        for label_id in range(len(settings.EMOTION_LABELS))
    }
    print(f"[train_emotion] 训练集规模: {len(train_texts)}，分布: {label_distribution}")
    print(f"[train_emotion] 验证集规模: {len(val_texts)}")

    # ===== 模型初始化：基座 + 四分类头 =====
    id2label = {idx: label for idx, label in enumerate(settings.EMOTION_LABELS)}
    label2id = {label: idx for idx, label in enumerate(settings.EMOTION_LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(settings.EMOTION_LABELS),
        id2label=id2label,   # 标签映射写入 config.json，推理端零额外约定
        label2id=label2id,
    )

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

    # 类别权重随模型置于同一设备（CPU 训练 / GPU 训练均兼容）
    device = next(model.parameters()).device
    class_weights = torch.tensor(
        [CLASS_WEIGHTS[label] for label in settings.EMOTION_LABELS],
        dtype=torch.float32,
    ).to(device)

    output_dir = os.path.join("outputs", "emotion_training")
    trainer = WeightedTrainer(
        model=model,
        class_weights=class_weights,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,             # 全量微调的标准学习率
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            # 以高危召回率为最优权重选择依据（安全底线优先于整体精度）
            metric_for_best_model="high_risk_recall",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_total_limit=2,
            logging_steps=50,
        ),
        # 早停：高危召回率连续 2 个 epoch 无提升即终止训练
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ===== 保存最终权重到配置约定路径，供 EmotionClassifier 直接加载 =====
    final_path = settings.EMOTION_MODEL_PATH
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[train_emotion] 模型已保存至 {final_path}")
    print("[train_emotion] 下一步：运行 python evaluation/emotion_eval.py 验证高危召回率 > 95%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLM-RoBERTa 情绪分类模型微调")
    parser.add_argument("--train", default="data/eval/emotion_train.json",
                        help="训练语料路径（JSON 数组，含 text/label 字段）")
    parser.add_argument("--val", default=None,
                        help="验证语料路径，缺省时从训练集分层切出 20%%")
    args = parser.parse_args()

    train(args.train, args.val)
