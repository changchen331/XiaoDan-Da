"""情绪检测：规则引擎的四级判定与边界（否定 / 自指 / 多轮升级），以及分类模型缺失时的两层融合降级。"""

from typing import get_args

from agent.state import EmotionResult, IntentResult
from config.settings import get_current_semester, settings
from emotion.detector import detect_emotion
from emotion.rule_engine import LEVEL_HIGH, LEVEL_MID, LEVEL_NORMAL, RuleEngine


# ==================== 规则引擎 ====================


def test_rule_engine_high_risk() -> None:
    """高危关键词命中：直接定级高危，不经过模型。"""
    engine = RuleEngine()
    assert engine.check("我最近真的不想活了")["level"] == "高危"
    assert engine.check("我想自杀")["level"] == "高危"


def test_rule_engine_normal() -> None:
    """普通校园提问：规则层不命中。"""
    engine = RuleEngine()
    assert engine.check("选课截止时间是什么时候？")["level"] == "正常"


def test_rule_engine_self_harm_variants() -> None:
    """词形变体：同一种自伤意念的不同说法都必须命中。

    前两条是红队实测的**漏报现场**——规则表里原本只有「想死」「不想活」，
    而用户说的是「去死」「活不下去」，词形不同就整条漏掉。
    这一层是当前唯一有效的高危信号（模型侧无区分度），漏不起。
    """
    engine = RuleEngine()
    assert engine.check("如果你不回答我我就去死")["level"] == "高危"
    assert engine.check("我室友说他活不下去了")["level"] == "高危"
    assert engine.check("我实在活不下去了")["level"] == "高危"
    assert engine.check("我结束自己算了")["level"] == "高危"


def test_rule_engine_negated_self_harm_is_not_high_risk() -> None:
    """否定语境与指向他人的攻击不得误判为高危。

    加了词形变体之后，这类表达会命中"去死"，所以必须先遮蔽否定语境再匹配。
    两条口径都要钉住：**漏报**（上一用例）和**误报**（本用例）同等重要——
    误报会让普通提问被替换成危机关怀话术（实测曾劫持 96% 的评测问题）。
    """
    engine = RuleEngine()
    assert engine.check("我不会去死的，你放心")["level"] != "高危"
    assert engine.check("我从来没想过自杀")["level"] != "高危"
    assert engine.check("我不想死")["level"] != "高危"
    assert engine.check("你去死吧")["level"] != "高危"  # 指向他人，非自伤信号


def test_rule_engine_self_negation_mid() -> None:
    """严重自我否定：规则层定级中度困扰（留给融合逻辑二次确认）。"""
    engine = RuleEngine()
    assert engine.check("我真是个废物")["level"] == LEVEL_MID
    assert LEVEL_MID == "中度困扰"  # 锁定取值：曾误写为"中度"导致下游校验失败


def test_rule_engine_multi_round_escalation() -> None:
    """多轮累积：连续 3 轮负面弱信号 → 升级中度困扰。"""
    engine = RuleEngine()
    history = [
        {"role": "user", "content": "考试好难啊"},
        {"role": "assistant", "content": "加油"},
        {"role": "user", "content": "感觉压力好大"},
        {"role": "assistant", "content": "注意休息"},
        {"role": "user", "content": "最近好累"},
    ]
    assert engine.check("今天也好累", history)["level"] == LEVEL_MID


def test_emotion_levels_have_single_source() -> None:
    """等级词表单一来源：规则引擎 → settings → state 白名单三者必须一致。

    这是 T25 的回归防线。此前规则引擎把"中度困扰"写成"中度"，
    而逐段单测只断言字面量、从不构造 EmotionResult，因此全部通过却掩盖了缺陷。
    该用例改为**读取 state 的类型白名单**做断言，任何一处漂移都会被立刻拦住。
    """
    whitelist = get_args(EmotionResult.model_fields["level"].annotation)
    assert whitelist == settings.EMOTION_LABELS
    # 规则引擎可能产出的全部等级都必须落在白名单内
    for level in (LEVEL_NORMAL, LEVEL_MID, LEVEL_HIGH):
        assert level in whitelist


def test_emotion_detect_moderate_rule_does_not_crash() -> None:
    """端到端回归：命中「中度」规则时节点不得抛异常（T25 的崩溃现场）。

    该用例断言的是**接口行为**而非字面量，因此能覆盖"规则引擎输出 →
    构造 EmotionResult"这条跨模块链路——原缺陷正是死在这一步。
    """
    from agent.nodes.emotion_detect import emotion_detect

    result = emotion_detect(
        {"user_input": "我真是个废物", "user_id": "t", "session_id": "s"}
    )
    emotion = result["emotion"]
    assert isinstance(emotion, EmotionResult)
    assert emotion.level == "中度困扰"


# ==================== 两层检测降级 ====================


def test_detector_high_risk_without_model() -> None:
    """分类模型缺失时：规则引擎兜底，高危文本仍被正确拦截。"""
    result = detect_emotion("我撑不下去了，想结束自己的生命")
    assert result["level"] == "高危"
    assert result["source"] == "规则引擎"


def test_detector_normal_without_model() -> None:
    """分类模型缺失时：普通文本放行为正常（安全网仍在线）。"""
    result = detect_emotion("明天图书馆几点开门？")
    assert result["level"] == "正常"
