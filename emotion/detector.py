"""情绪检测引擎 - 两层融合入口。

检测流程（对应架构文档模块二）：
1. 第一层规则引擎：零延迟（纯正则匹配，<1ms）拦截最明显的危险信号，
   高危规则命中直接定级，不再经过模型 —— 这是安全兜底网，
   确保即使分类模型遇到训练集未覆盖的表达，最危险的 case 也绝不漏掉
2. 第二层分类模型：XLM-RoBERTa 语义理解，覆盖规则无法触达的隐晦表达
3. 融合升级：规则判"中度" + 模型高危概率超阈值 → 升级为"高危"
   （两层同时给出中等风险信号时，按"宁严勿漏"原则取高）

性能设计：分类器为进程级单例（模型加载耗时数秒，绝不能每条消息重载）。
"""
from config.settings import settings
from emotion.rule_engine import RuleEngine

# 进程级单例：规则引擎（纯正则，构造开销小，但单例避免重复编译正则）
_rule_engine: "RuleEngine | None" = None
# 进程级单例：分类器（模型加载耗时，仅初始化一次）
_classifier = None
# 分类器可用性标记：首次加载失败后置 False，避免每条消息重复尝试加载
_classifier_unavailable = False


def detect_emotion(text: str, conversation_history: list | None = None) -> dict:
    """对用户输入执行两层情绪检测。

    :param text: 用户当前输入
    :param conversation_history: 最近对话历史 [{"role", "content"}, ...]，
        供规则引擎做多轮累积检测
    :return: {"level", "source", "confidence"}
        - level: 正常 / 轻度困扰 / 中度困扰 / 高危
        - source: 规则引擎 / 分类模型 / 规则+模型融合
        - confidence: 置信度 0-1
    """
    global _rule_engine

    if _rule_engine is None:
        _rule_engine = RuleEngine()

    # ===== 第一层：规则引擎（安全兜底网，高危即阻断）=====
    rule_result = _rule_engine.check(text, conversation_history)
    if rule_result["level"] == "高危":
        return {"level": "高危", "source": "规则引擎", "confidence": 1.0}

    # ===== 第二层：分类模型（语义级检测）=====
    model_probs = _predict_with_model(text)

    if model_probs is None:
        # 模型不可用（权重缺失 / 依赖异常）时的降级策略：
        # 规则中度维持中度，其余放行为正常，安全兜底网仍在
        level = rule_result["level"] if rule_result["level"] == "中度" else "正常"
        return {"level": level, "source": "规则引擎", "confidence": 0.6}

    model_level = max(model_probs, key=model_probs.get)

    # ===== 融合升级：规则中度 + 模型高危概率超阈值 → 高危 =====
    if (rule_result["level"] == "中度"
            and model_probs.get("高危", 0.0) > settings.HIGH_RISK_PROB_THRESHOLD):
        return {"level": "高危", "source": "规则+模型融合", "confidence": model_probs["高危"]}

    # ===== 模型独立判断（正常 / 轻度 / 中度）=====
    return {"level": model_level, "source": "分类模型", "confidence": model_probs[model_level]}


def _predict_with_model(text: str) -> dict | None:
    """调用分类模型，模型不可用时返回 None（降级为纯规则引擎）。

    首次加载失败后置不可用标记，后续消息直接走规则引擎，
    避免每条消息都触发一次注定失败的模型加载。
    """
    global _classifier, _classifier_unavailable

    if _classifier_unavailable:
        return None

    try:
        if _classifier is None:
            from emotion.classifier import EmotionClassifier

            _classifier = EmotionClassifier()
        return _classifier.predict(text)
    except Exception as load_error:
        # 模型路径不存在 / transformers 依赖异常等：打印日志并永久降级
        _classifier_unavailable = True
        print(f"[detector] 分类模型不可用，本进程降级为纯规则引擎: {load_error}")
        return None
