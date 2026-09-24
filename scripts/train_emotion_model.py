"""XLM-RoBERTa 情绪分类模型微调脚本（全量微调，单卡 4090 约 25-50 分钟）。

用法：
    python scripts/train_emotion_model.py [--train 路径] [--val 路径]

数据格式（JSON 数组，UTF-8）：
    [{"text": "最近考试压力好大", "label": "轻度困扰"}, ...]
标签取值：正常 / 轻度困扰 / 中度困扰 / 高危

数据构成（对应架构文档模块二）：
- 公开数据集：NLPCC 微博情绪 / ChnSentiCorp / 抑郁倾向对话（基础覆盖）
- 校园场景定制语料：心理学专业编写 + LLM 合成的隐晦表达（核心价值所在）

类别不平衡处理（两级矫正，倍率随语料实际分布自适应）：
- 损失层：逆频次 class weight，强迫模型为少数类的错误多付梯度
- 数据层：高危样本 2 倍过采样，放大其在每个 epoch 中的出现频率

权重**不写死倍率**的理由（v2 实测教训）：原本写死"高危 ×3"，
其前提是真实分布中高危不足 8%（相对均匀占比 25% 属严重稀缺）。
但 v2 的闭环语料刻意把高危提到 30%，此时 ×3 叠加 2 倍过采样会让高危
占据约 55% 的损失质量，模型直接退化成"全部判高危"——
高危召回率 100%、精确率只有 30%，硬性指标看着达标而系统实际不可用。
改成逆频次后，高危占 8% 时权重约 3.1（与原设定一致），
占 30% 时约 0.87（不再过度矫正），两种语料分布下各类损失质量都接近均匀。

早停策略：监控验证集高危 F1（而非整体准确率——
整体准确率会被占多数的"正常"类主导，无法反映安全底线），
连续 2 个 epoch 无提升即停止，恢复最优权重。
不用高危召回率做早停/选优的理由见 compute_metrics。
"""

import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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
    """带类别权重的 Trainer：损失按**逆频次权重**加权（倍率由语料分布算出，不写死）。

    原生 Trainer 使用等权 CrossEntropy，本子类只覆写 compute_loss 一处，
    其余训练逻辑（梯度累积 / 混合精度 / 日志）全部复用父类实现。
    权重从何而来见 `inverse_frequency_weights` 与模块 docstring 的"v2 实测教训"。
    """

    def __init__(self, *args, class_weights=None, **kwargs) -> None:
        """初始化并持有类别权重张量。

        :param class_weights: 形状 [num_labels] 的权重张量（**留在 CPU**，
            见 compute_loss 中的设备对齐说明）
        """
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """加权损失计算：labels 从 inputs 中取出单独参与损失函数。

        权重张量在此处才对齐到 logits 所在设备。模型是在 Trainer 内部
        才被搬到 GPU 的，若在构造 Trainer 时就把权重固定到某个设备，
        CPU 上跑得通、换到 CUDA 就会报
        "Expected all tensors to be on the same device, but got weight is on cpu"。
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        weight = self.loss_fct.weight
        if weight is not None and weight.device != outputs.logits.device:
            self.loss_fct.weight = weight.to(outputs.logits.device)

        loss = self.loss_fct(
            outputs.logits.view(-1, model.config.num_labels), labels.view(-1)
        )
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
    打乱用固定种子的局部随机源：全局 random 的种子随进程而变，
    会让同一份语料训出不同的模型，断掉"同样的输入得到同样的结果"这条底线。
    """
    high_risk_id = settings.EMOTION_LABELS.index("高危")
    pairs = list(zip(texts, labels))
    high_risk_pairs = [pair for pair in pairs if pair[1] == high_risk_id]

    oversampled = pairs + high_risk_pairs * (HIGH_RISK_OVERSAMPLE - 1)
    random.Random(42).shuffle(oversampled)  # 打乱防止 batch 内类别聚集
    return [text for text, _ in oversampled], [label for _, label in oversampled]


def inverse_frequency_weights(labels: list, num_labels: int) -> list:
    """按实际频次计算类别权重：weight_i = N / (K × n_i)。

    该式让每个类别在损失中的总质量相等（n_i × weight_i = N/K），
    因此无论语料把高危放在 8% 还是 30%，都不会出现某一类压倒其余类的情况。
    若某类在语料中完全缺席，按计数 1 处理，避免除零。

    :param labels: 训练集的类别 id 列表（过采样后）
    :param num_labels: 类别总数
    :return: 形状 [num_labels] 的权重列表
    """
    total = len(labels)
    counts = [max(labels.count(index), 1) for index in range(num_labels)]
    return [total / (num_labels * count) for count in counts]


def compute_metrics(eval_prediction) -> dict:
    """验证指标：整体准确率 + 高危召回率 + 高危 F1。

    高危召回率 = 高危样本中被正确识别为高危的比例，
    是本任务的硬性上线指标（> 95%），漏报代价远高于误报。

    但召回率**不能用作模型选择标准**（v2 实测教训）：
    "把所有样本都判成高危"这一退化解会让召回率直接等于 1.0，
    以它选最优权重，训练器必然挑中那个退化模型——
    实测两次退化都发生在 epoch 1，且都被 metric_for_best_model 选中。
    因此模型选择改用高危 F1（精确率与召回率的调和平均），
    退化解在此指标上得 0.46，而正常模型约 0.6，方向正确。
    硬性上线门槛仍只看召回率（漏报一条真危机的代价远高于多报一条）。
    """
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids

    high_risk_id = settings.EMOTION_LABELS.index("高危")
    high_risk_recall = recall_score(
        labels,
        predictions,
        labels=[high_risk_id],
        average=None,
        zero_division=0,
    )[0]
    high_risk_precision = precision_score(
        labels,
        predictions,
        labels=[high_risk_id],
        average=None,
        zero_division=0,
    )[0]
    high_risk_f1 = f1_score(
        labels,
        predictions,
        labels=[high_risk_id],
        average=None,
        zero_division=0,
    )[0]

    return {
        "accuracy": accuracy_score(labels, predictions),
        "high_risk_recall": high_risk_recall,
        "high_risk_precision": high_risk_precision,
        "high_risk_f1": high_risk_f1,
    }


def train(train_path: str, val_path: str | None, epochs: int) -> None:
    """微调主流程：加载语料 → 过采样 → 训练（加权损失）→ 保存。

    验证集的取舍（v2 实测教训）：语料只有 100 条时，切出 20 条做验证，
    其中高危仅 6 条——任何基于它的"最优权重选择"都是噪声驱动的。
    实测同一份语料三种配置分别落到三个不同的退化点（全判高危 / 55.6% 召回 /
    全判正常），说明瓶颈不在超参而在语料规模。

    因此默认（不传 --val）改为：**全量语料训练、固定轮数、不做早停与选优**，
    把 100 条样本全部用于学习，诚实的效果报告交给独立的 60 条评测集
    （evaluation/emotion_eval.py）。语料扩到 1500 条（v3）后，
    传 --val 即可恢复"留出验证集 + 早停 + 选优"的常规流程。

    :param train_path: 训练语料路径
    :param val_path: 独立验证语料路径；提供时启用早停与最优权重选择
    :param epochs: 训练轮数（提供 --val 时为最大轮数，早停可提前结束）
    """
    import torch

    texts, labels = load_corpus(train_path)

    if val_path:
        val_texts, val_labels = load_corpus(val_path)
    else:
        # 全量训练：验证集与训练集同源，仅用于打印训练指标（不参与选优，
        # 因为下面的 load_best_model_at_end / EarlyStoppingCallback 都被关闭）
        val_texts, val_labels = list(texts), list(labels)

    # 高危过采样（仅训练集）
    train_texts, train_labels = oversample_high_risk(texts, labels)
    label_distribution = {
        settings.EMOTION_LABELS[label_id]: train_labels.count(label_id)
        for label_id in range(len(settings.EMOTION_LABELS))
    }
    print(f"[train_emotion] 训练集规模: {len(train_texts)}，分布: {label_distribution}")
    print(
        f"[train_emotion] 验证集规模: {len(val_texts)}（{'独立验证集' if val_path else '无独立验证集，仅打印指标'}）"
    )

    # ===== 模型初始化：基座 + 四分类头 =====
    id2label = {idx: label for idx, label in enumerate(settings.EMOTION_LABELS)}
    label2id = {label: idx for idx, label in enumerate(settings.EMOTION_LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(settings.EMOTION_LABELS),
        id2label=id2label,  # 标签映射写入 config.json，推理端零额外约定
        label2id=label2id,
    )

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

    # 类别权重：按过采样后的实际频次做逆频次矫正（理由见模块 docstring）；
    # 张量**保持默认的 CPU 放置**，由 WeightedTrainer.compute_loss 在每次前向时
    # 对齐到 logits 所在设备（模型由 Trainer 搬到 GPU，
    # 构造阶段固定设备会在 CUDA 训练时报 device mismatch）
    weights = inverse_frequency_weights(train_labels, len(settings.EMOTION_LABELS))
    print(
        "[train_emotion] 类别权重: "
        + ", ".join(
            f"{label}={weight:.2f}"
            for label, weight in zip(settings.EMOTION_LABELS, weights)
        )
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)

    output_dir = os.path.join("outputs", "emotion_training")
    trainer = WeightedTrainer(
        model=model,
        class_weights=class_weights,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,  # 全量微调的标准学习率
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            # 选优与早停**只在有独立验证集时启用**：无验证集时 eval_dataset 就是
            # 训练集本身，靠它挑权重等于按训练集挑，只会挑到过拟合最重的那一轮
            metric_for_best_model="high_risk_f1",
            greater_is_better=True,
            load_best_model_at_end=bool(val_path),
            save_total_limit=2,
            logging_steps=50,
        ),
        # 早停同样只在有独立验证集时启用（判据：高危 F1 连续 2 轮无提升）
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=2)] if val_path else []
        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ===== 保存最终权重到配置约定路径，供 EmotionClassifier 直接加载 =====
    final_path = settings.EMOTION_MODEL_PATH
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[train_emotion] 模型已保存至 {final_path}")
    print(
        "[train_emotion] 下一步：运行 python evaluation/emotion_eval.py 验证高危召回率 > 95%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLM-RoBERTa 情绪分类模型微调")
    parser.add_argument(
        "--train",
        default="data/eval/emotion_train.json",
        help="训练语料路径（JSON 数组，含 text/label 字段）",
    )
    parser.add_argument(
        "--val",
        default=None,
        help="验证语料路径；**缺省为全量训练**：不切验证集、不早停、不选优，只打印指标",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="最大训练轮数；用于对照「召回率低是语料不足还是训练不充分」",
    )
    args = parser.parse_args()

    train(args.train, args.val, args.epochs)
