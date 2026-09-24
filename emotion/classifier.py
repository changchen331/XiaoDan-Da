"""情绪检测引擎 - 第二层：XLM-RoBERTa 分类模型（语义级检测）。

职责：处理规则引擎无法覆盖的隐晦表达（如"我最近总是很开心地加班到凌晨三点"
这类反讽 / 曲线表达），输出四级概率分布。

模型：XLM-RoBERTa-base 全量微调（270M 参数，CPU 推理约 30ms、GPU 约 5-10ms），
跨语言预训练使其同时覆盖中英文情绪表达（留学生场景）。

标签约定（与训练脚本一致，顺序即类别 id）：
    0=正常  1=轻度困扰  2=中度困扰  3=高危
"""

import torch

from config.settings import settings


class EmotionClassifier:
    """微调后的 XLM-RoBERTa 情绪分类器。"""

    def __init__(self) -> None:
        """加载微调后的模型权重。

        路径固定取 settings.EMOTION_MODEL_PATH（由 scripts/train_emotion_model.py
        训练产出，含 config.json）。此前这里有个 `model_path` 参数，
        但全仓没有任何调用点传它——留着只会让读者以为存在"多模型切换"能力。

        :raises OSError: 模型路径不存在时由 transformers 抛出，
            由 detector 层捕获并降级为纯规则引擎
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = settings.EMOTION_MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        # 标签顺序与训练时保持一致，索引即模型输出 logits 的列号
        self.labels: list = list(settings.EMOTION_LABELS)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """输出四级情绪概率分布。

        :param text: 用户输入文本（截断至 128 token，
            情绪表达集中在前段，长文本截断不影响判断）
        :return: {"正常": p0, "轻度困扰": p1, "中度困扰": p2, "高危": p3}，
            四项概率之和为 1
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return dict(zip(self.labels, probs))
