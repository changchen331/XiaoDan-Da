"""情绪检测引擎 - 三层融合入口。

检测流程（对应架构文档模块二）：
1. 第一层规则引擎：零延迟（纯正则匹配，<1ms）拦截最明显的危险信号，
   高危规则命中直接定级，不再经过模型 —— 这是安全兜底网，
   确保即使后续各层都故障，最危险的 case 也绝不漏掉
2. 第二层分类模型：XLM-RoBERTa 语义理解，覆盖规则无法触达的隐晦表达
   （**只有模型最高概率超过 `HIGH_RISK_PROB_THRESHOLD` 时才采纳其结论**：
   低于该值时四类输出近乎均匀，argmax 只是噪声里的最大值——
   照单全收会既驳回规则引擎的中度判断，又把整条问答链路劫持成危机关怀，见下）
3. 第三层语义判别（`emotion/semantic.py`）：LLM 事实抽取，**只对规则未命中高危
   的消息**做复核，补上规则在隐晦表达上的漏报（评测集 18 条隐晦高危命中 0 条）。
   它在有事实依据（指向=本人）时才**升级**为高危；自身故障时维持前两层结论。

**为什么"降级"不在这一层**：规则命中高危时第一层已直接返回，不再走后续各层——
显式高危的响应必须零延迟、且不依赖任何外部服务。代价是"代他人求助"这类
规则误报**不会被本层下调**：该标注改由上报环节在**后台**补（见
`agent/nodes/report.py`），既不改定级、也不占用用户路径。

性能设计：分类器为进程级单例（模型加载耗时数秒，绝不能每条消息重载）。
"""

import threading

from agent.llm_clients import LLMUnavailableError
from config.settings import settings
from emotion.rule_engine import LEVEL_HIGH, LEVEL_MID, LEVEL_NORMAL, RuleEngine
from emotion.semantic import REFERENT_SELF, judge_semantic_stable

# 进程级单例：规则引擎（纯正则，构造开销小，但单例避免重复编译正则）
_rule_engine: "RuleEngine | None" = None
# 进程级单例：分类器（模型加载耗时，仅初始化一次）
_classifier = None
# 分类器可用性标记：首次加载失败后置 False，避免每条消息重复尝试加载
_classifier_unavailable = False
# 分类器初始化锁：初始化不是原子操作（导入 transformers + 加载权重耗时数秒），
# 并发首请求下多个线程会同时进入构造，实测其中一个会拿到**半初始化的
# transformers** 并抛 ImportError；而异常分支会把 _classifier_unavailable 置真，
# 等于**整个进程此后永久降级为纯规则引擎**（串行调用不复现，故极易漏测）
_classifier_lock = threading.Lock()


def detect_emotion(text: str, conversation_history: list | None = None) -> dict:
    """对用户输入执行三层情绪检测。

    :param text: 用户当前输入
    :param conversation_history: 最近对话历史 [{"role", "content"}, ...]，
        供规则引擎做多轮累积检测
    :return: {"level", "source", "confidence", "referent"}
        - level: 正常 / 轻度困扰 / 中度困扰 / 高危
        - source: 规则引擎 / 分类模型 / 规则+模型融合 / 语义判别
        - confidence: 置信度 0-1
        - referent: 危险表达的指向（本人/他人/引用/否定/无），未判定时为 None
    """
    global _rule_engine

    if _rule_engine is None:
        _rule_engine = RuleEngine()

    # ===== 第一层：规则引擎（安全兜底网，高危即阻断）=====
    rule_result = _rule_engine.check(text, conversation_history)
    if rule_result["level"] == LEVEL_HIGH:
        return _result(LEVEL_HIGH, "规则引擎", 1.0)

    # ===== 第二层：分类模型（语义级检测）=====
    baseline = _classify_with_model(text, rule_result)

    # ===== 第三层：语义判别（只做"有依据的升级"）=====
    if not settings.EMOTION_SEMANTIC_ENABLED:
        return baseline
    return _apply_semantic(text, baseline)


def _classify_with_model(text: str, rule_result: dict) -> dict:
    """第二层：分类模型判定（模型不可用或无区分度时以规则结论为准）。"""
    model_probs = _predict_with_model(text)

    if model_probs is None or max(model_probs.values()) <= (
        settings.HIGH_RISK_PROB_THRESHOLD
    ):
        # 模型不可用，或可用但**输出不构成证据**时，一律以规则引擎为准。
        #
        # 判据就是 HIGH_RISK_PROB_THRESHOLD：模型最高概率没超过它，
        # 说明四类输出近乎均匀，argmax 只是在噪声里挑了个最大的。
        # 实测（100 条语料训出的模型）："我真是个废物"得到
        # {正常 0.273, 轻度 0.260, 中度 0.193, 高危 0.275}——最高与最低只差 0.08，
        # argmax 恰好落在"高危"上。若照单全收会有两个后果：
        # ① 用噪声驳回规则引擎明确的中度判断，安全网被绕过；
        # ② 100 条普通事务性提问里 95 条被 argmax 判为高危（高危概率从未超过 0.30），
        #    整条问答链路被劫持——实测 100 条评测问题有 96 条根本没走到检索，
        #    回答被替换成危机关怀话术。
        # 只有 max 超过阈值时，模型的结论才进入后续判定；
        # 那时 argmax 落在高危上也才等价于"高危概率足够高"。
        #
        # 降级策略：规则中度维持中度，其余放行为正常，安全兜底网仍在
        level = (
            rule_result["level"] if rule_result["level"] == LEVEL_MID else LEVEL_NORMAL
        )
        return _result(level, "规则引擎", 0.6)

    model_level = max(model_probs, key=model_probs.get)

    # ===== 融合升级：规则中度 + 模型高危概率超阈值 → 高危 =====
    if (
        rule_result["level"] == LEVEL_MID
        and model_probs.get(LEVEL_HIGH, 0.0) > settings.HIGH_RISK_PROB_THRESHOLD
    ):
        return _result(LEVEL_HIGH, "规则+模型融合", model_probs[LEVEL_HIGH])

    # ===== 模型独立判断（正常 / 轻度 / 中度）=====
    return _result(model_level, "分类模型", model_probs[model_level])


def _apply_semantic(text: str, baseline: dict) -> dict:
    """第三层：语义判别复核——只在**有事实依据**时升级为高危。

    升级条件必须同时满足两条，缺一不可：
    1. 语义层判「高危」；
    2. 指向为「本人」——若危险表达指向他人 / 引用 / 否定（有事实依据表明
       危险不属于提问者本人），则**不升级**。这正是"误报仅在有事实依据时处理"
       的落地方式：语义层的高危结论本身也要经得起指向的检验。

    语义层故障（降级链耗尽 / 输出非法）时**维持基线判决**——增强层故障
    不改变基线结论，与 `faq_verify` 的"信任阈值"同构。
    """
    try:
        verdict = judge_semantic_stable(text)
    except (LLMUnavailableError, ValueError, KeyError, TypeError) as semantic_error:
        print(f"[detector] 语义判别不可用，维持前两层结论: {semantic_error}")
        return baseline

    referent = verdict["referent"]
    if verdict["level"] == LEVEL_HIGH and referent == REFERENT_SELF:
        return _result(LEVEL_HIGH, "语义判别", 0.9, referent)

    # 未升级：把指向作为标注带回（供上报记录区分"本人危机 / 代他人求助"），
    # 等级仍沿用前两层结论
    return {**baseline, "referent": referent}


def _result(
    level: str, source: str, confidence: float, referent: str | None = None
) -> dict:
    """构造检测结果（统一字段，避免各处漏写 referent）。"""
    return {
        "level": level,
        "source": source,
        "confidence": confidence,
        "referent": referent,
    }


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
            # 双重检查：等锁期间可能已被先到的线程初始化完成，不必再构造一次
            with _classifier_lock:
                if _classifier is None:
                    from emotion.classifier import EmotionClassifier

                    _classifier = EmotionClassifier()
        return _classifier.predict(text)
    except Exception as load_error:
        # 模型路径不存在 / transformers 依赖异常等：打印日志并永久降级
        _classifier_unavailable = True
        print(f"[detector] 分类模型不可用，本进程降级为纯规则引擎: {load_error}")
        return None
