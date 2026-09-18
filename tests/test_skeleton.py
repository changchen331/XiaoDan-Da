"""核心逻辑单元测试：不依赖任何外部服务（LLM / 数据库 / 向量库），
覆盖规则引擎、切分策略、文本清洗、两层降级、学期计算与 JSON 解析、
FAQ 校验信任策略、关怀后缀分级、条件路由与意图兜底。

运行：python -m pytest tests/test_skeleton.py -v
"""

from datetime import date
from typing import get_args

import pytest

from agent.llm_clients import parse_json_response
from agent.state import EmotionResult, IntentResult
from config.settings import get_current_semester, settings
from emotion.detector import detect_emotion
from emotion.rule_engine import LEVEL_HIGH, LEVEL_MID, LEVEL_NORMAL, RuleEngine
from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.cleaner import clean_text


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

    result = emotion_detect({"user_input": "我真是个废物", "user_id": "t", "session_id": "s"})
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


# ==================== 切分策略 ====================


def test_chunker_faq_pairs() -> None:
    """FAQ 切分：按问答对边界切，每对一个 chunk。"""
    text = "问：校园卡丢了怎么办？\n答：请到一卡通中心挂失补办。\n问：图书馆几点开门？\n答：早8点。"
    chunks = chunk_by_type(text, "FAQ")
    assert len(chunks) == 2
    assert "校园卡" in chunks[0]["text"]
    assert chunks[0]["metadata"]["doc_type"] == "FAQ"


def test_chunker_notice_by_title() -> None:
    """通知切分：按标题行分界，每条通知一个 chunk。"""
    text = (
        "关于2026年选课安排的通知\n第一条内容。\n\n" "关于宿舍调整的通知\n第二条内容。"
    )
    chunks = chunk_by_type(text, "通知")
    assert len(chunks) == 2


def test_chunker_manual_splits_long_text() -> None:
    """手册切分：长文本被递归切为多个 chunk 且带重叠。"""
    long_text = "\n\n".join(
        f"第{i}段落内容。" + "课程设置的详细说明。" * 30 for i in range(20)
    )
    chunks = chunk_by_type(long_text, "手册")
    assert len(chunks) > 1
    assert all(chunk["metadata"]["doc_type"] == "手册" for chunk in chunks)


# ==================== 文本清洗 ====================


def test_cleaner_removes_noise() -> None:
    """清洗：页脚版权、备案号、分享残留等噪声行被去除。"""
    text = "复旦大学选课须知\n正文第一段。\n版权所有 © 复旦大学\n沪ICP备00000001号\n正文第二段。"
    cleaned = clean_text(text)
    assert "版权所有" not in cleaned
    assert "沪ICP" not in cleaned
    assert "正文第一段" in cleaned


# ==================== 工具函数 ====================


def test_get_current_semester() -> None:
    """学期计算：9 月起为秋季，2 月为春季，7 月为暑期。"""
    assert get_current_semester(date(2026, 9, 8)) == "2026-2027秋季"
    assert get_current_semester(date(2027, 1, 15)) == "2026-2027秋季"
    assert get_current_semester(date(2026, 2, 20)) == "2025-2026春季"
    assert get_current_semester(date(2026, 7, 10)) == "2025-2026暑期"


def test_parse_json_response_with_wrapper() -> None:
    """JSON 解析：容忍代码块包裹与前后说明文字。"""
    raw = '好的，结果如下：\n```json\n{"category": "FAQ", "rewritten_query": "校园卡挂失"}\n```'
    parsed = parse_json_response(raw)
    assert parsed["category"] == "FAQ"
    assert parsed["rewritten_query"] == "校园卡挂失"


def test_parse_json_response_plain() -> None:
    """JSON 解析：标准 JSON 直接解析。"""
    parsed = parse_json_response('{"passed": true, "score": 0.9}')
    assert parsed["passed"] is True
    assert parsed["score"] == 0.9


# ==================== 状态结构 ====================


def test_agent_state_fields() -> None:
    """AgentState 字段完整性：与架构文档定义一致。"""
    from agent.state import AgentState

    expected = {
        "user_input",
        "user_id",
        "session_id",
        "emotion",
        "intent",
        "response_language",
        "retrieved_contexts",
        "faq_hit",
        "generated_response",
        "quality",
        "retry_count",
        "memory_summary",
        "final_response",
        "should_end",
    }
    assert expected.issubset(set(AgentState.__annotations__))


# ==================== FAQ 轻量校验 ====================


def test_faq_verify_trust_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """信任阈值策略：校验调用自身异常时放行（快路径不因增强层故障失效）。"""
    import agent.faq_verify as faq_verify_module

    def _broken_call(prompt: str, system: str | None = None) -> str:
        raise ValueError("模拟端点不可达")

    monkeypatch.setattr(faq_verify_module, "chat_qwen_json", _broken_call)
    assert (
        faq_verify_module.verify_faq_match(
            "校园卡丢了怎么办", "校园卡挂失流程", "请到一卡通中心挂失。", "中文"
        )
        is True
    )


def test_faq_verify_rejects_answer_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """答案不适配被拒绝：问 A 答 B 时不得直返标准答案。"""
    import agent.faq_verify as faq_verify_module

    monkeypatch.setattr(
        faq_verify_module,
        "chat_qwen_json",
        lambda prompt, system=None: '{"answer_match": false, "language_match": true}',
    )
    assert (
        faq_verify_module.verify_faq_match(
            "校园卡补办收费吗", "校园卡挂失流程", "请到一卡通中心挂失。", "中文"
        )
        is False
    )


def test_faq_verify_rejects_language_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """语言不匹配被拒绝：中文标准答案不直返给期望英文的用户。"""
    import agent.faq_verify as faq_verify_module

    monkeypatch.setattr(
        faq_verify_module,
        "chat_qwen_json",
        lambda prompt, system=None: '{"answer_match": true, "language_match": false}',
    )
    assert (
        faq_verify_module.verify_faq_match(
            "How do I report a lost campus card?",
            "校园卡挂失流程",
            "请到一卡通中心挂失。",
            "English",
        )
        is False
    )


# ==================== 关怀后缀分级 ====================


def _make_state(
    emotion_level: str,
    response_language: str = "中文",
    generated: str = "选课截止时间为 9 月 15 日。",
) -> dict:
    """构造 care_suffix 单测所需的最小 State。"""
    return {
        "user_input": "选课截止时间是什么时候？",
        "generated_response": generated,
        "emotion": EmotionResult(
            level=emotion_level, source="分类模型", confidence=0.9
        ),
        "response_language": response_language,
    }


def test_care_suffix_normal_passthrough() -> None:
    """正常情绪：回复原样透传，零额外内容。"""
    from agent.nodes.care_suffix import care_suffix

    state = _make_state("正常")
    result = care_suffix(state)
    assert result["final_response"] == "选课截止时间为 9 月 15 日。"


def test_care_suffix_mild_fallback_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """轻度困扰 + LLM 失败：降级中文固定模板（关怀后缀不因模型故障缺失）。"""
    import importlib

    # importlib 返回 sys.modules 的真实模块对象：nodes/__init__ 的
    # from-import 会把同名属性遮蔽为节点函数，import as 拿不到模块
    care_suffix_module = importlib.import_module("agent.nodes.care_suffix")

    def _broken_call(
        prompt: str, system: str | None = None, temperature: float = 0.1
    ) -> str:
        raise ValueError("模拟轻量端点不可达")

    monkeypatch.setattr(care_suffix_module, "chat_qwen", _broken_call)
    result = care_suffix_module.care_suffix(_make_state("轻度困扰"))
    assert result["final_response"].startswith("选课截止时间为 9 月 15 日。")
    assert "注意休息" in result["final_response"]


def test_care_suffix_moderate_fallback_contains_care_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中度困扰 + LLM 失败：降级模板包含关怀与渠道信息（联系方式非模型编造）。"""
    import importlib

    care_suffix_module = importlib.import_module("agent.nodes.care_suffix")

    def _broken_call(
        prompt: str, system: str | None = None, temperature: float = 0.1
    ) -> str:
        raise ValueError("模拟轻量端点不可达")

    monkeypatch.setattr(care_suffix_module, "chat_qwen", _broken_call)
    result = care_suffix_module.care_suffix(_make_state("中度困扰"))
    assert "心理咨询中心" in result["final_response"]


def test_care_suffix_appends_conversation_history() -> None:
    """care_suffix 追加本轮问答到 conversation_history（T26 回归）。

    该字段此前只有读取点、没有写入点，且 invoke 每轮重置它，导致多轮上下文恒为空。
    """
    from agent.nodes.care_suffix import care_suffix

    state = _make_state("正常")
    state["conversation_history"] = [{"role": "user", "content": "上一轮的问题"}]

    result = care_suffix(state)
    history = result["conversation_history"]

    assert history[0] == {"role": "user", "content": "上一轮的问题"}  # 旧历史保留
    assert history[-2] == {"role": "user", "content": state["user_input"]}
    assert history[-1] == {"role": "assistant", "content": result["final_response"]}


def test_care_suffix_history_is_capped() -> None:
    """历史长度受 HISTORY_MAX_ROUNDS 限制，不随轮次无限膨胀。"""
    from agent.nodes.care_suffix import care_suffix

    state = _make_state("正常")
    state["conversation_history"] = [
        {"role": "user", "content": f"旧消息 {i}"} for i in range(100)
    ]

    history = care_suffix(state)["conversation_history"]
    assert len(history) == settings.HISTORY_MAX_ROUNDS * 2
    assert history[-1]["role"] == "assistant"  # 截断后仍以本轮收尾


def test_care_response_appends_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高危分支不经 care_suffix，需自行追加历史（两条出口口径一致）。"""
    import importlib

    care_response_module = importlib.import_module("agent.nodes.care_response")
    monkeypatch.setattr(
        care_response_module, "chat_deepseek", lambda *args, **kwargs: "我在这里陪你。"
    )

    result = care_response_module.care_response({"user_input": "我不想活了"})
    history = result["conversation_history"]

    assert history[-2] == {"role": "user", "content": "我不想活了"}
    assert history[-1] == {"role": "assistant", "content": "我在这里陪你。"}


# ==================== 条件路由 ====================


def _intent_state(category: str) -> dict:
    """构造意图路由单测所需的最小 State。"""
    return {
        "intent": IntentResult(
            category=category, confidence=0.9, rewritten_query="测试查询"
        )
    }


def test_route_after_intent_merges_faq_into_retrieve() -> None:
    """意图分流：FAQ 与简单问答合流到 retrieve（快路径不依赖分类准确性）。"""
    from agent.nodes.intent_route import route_after_intent

    assert route_after_intent(_intent_state("简单问答")) == "retrieve"
    assert route_after_intent(_intent_state("FAQ")) == "retrieve"
    assert route_after_intent(_intent_state("复杂查询")) == "plan_and_retrieve"
    assert route_after_intent(_intent_state("闲聊越界")) == "chitchat_response"


def test_route_after_retrieve_by_faq_hit() -> None:
    """检索分流：FAQ 命中走关怀后缀直返，未命中进生成。"""
    from agent.nodes.retrieve import route_after_retrieve

    assert route_after_retrieve({"faq_hit": True}) == "care_suffix"
    assert route_after_retrieve({"faq_hit": False}) == "generate"


def test_route_after_care_suffix_chitchat_ends() -> None:
    """关怀后缀分流：闲聊不写长期记忆直接结束，其余写入。"""
    from langgraph.graph import END

    from agent.nodes.care_suffix import route_after_care_suffix

    assert route_after_care_suffix(_intent_state("闲聊越界")) == END
    assert route_after_care_suffix(_intent_state("简单问答")) == "memory_write"
    assert route_after_care_suffix({}) == END


# ==================== 意图路由兜底 ====================


def test_intent_route_fallback_keeps_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """路由失败兜底：按简单问答处理并沿用既有语言偏好（跨轮持久化不被异常打断）。"""
    import importlib

    intent_route_module = importlib.import_module("agent.nodes.intent_route")

    def _broken_call(prompt: str, system: str | None = None) -> str:
        raise ValueError("模拟 JSON 解析失败")

    monkeypatch.setattr(intent_route_module, "chat_qwen_json", _broken_call)
    state = {
        "user_input": "选课时间",
        "conversation_history": [],
        "user_profile": {"role": "本科生"},
        "response_language": "English",
    }
    result = intent_route_module.intent_route(state)
    assert result["intent"].category == "简单问答"
    assert result["intent"].rewritten_query == "选课时间"
    assert result["response_language"] == "English"


def test_intent_route_parses_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次会话：未指定过语言时，跟随当前输入的语言。"""
    import importlib

    intent_route_module = importlib.import_module("agent.nodes.intent_route")

    monkeypatch.setattr(
        intent_route_module,
        "chat_qwen_json",
        lambda prompt, system=None: '{"category": "FAQ", "rewritten_query": "校园卡挂失", '
        '"input_language": "English", "requested_language": ""}',
    )
    state = {
        "user_input": "How to report a lost card?",
        "conversation_history": [],
        "user_profile": {"role": "留学生"},
    }
    result = intent_route_module.intent_route(state)
    assert result["response_language"] == "English"
    assert result["intent"].category == "FAQ"


def test_intent_route_keeps_session_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """会话偏好持久化：本轮未提语言要求时，沿用既有的会话级偏好。

    回归用例——此前的实现把语言判定整体交给模型并"跟随当前输入语言"，
    导致用户说过的偏好被下一轮输入重置（实测：先要求英文，
    下一轮用中文提问后回复又变回中文）。
    """
    import importlib

    intent_route_module = importlib.import_module("agent.nodes.intent_route")

    monkeypatch.setattr(
        intent_route_module,
        "chat_qwen_json",
        lambda prompt, system=None: '{"category": "简单问答", "rewritten_query": "图书馆开放时间", '
        '"input_language": "中文", "requested_language": ""}',
    )
    state = {
        "user_input": "图书馆开放时间？",
        "conversation_history": [],
        "user_profile": {"role": "留学生"},
        "response_language": "English",  # 上一轮已确定
    }
    result = intent_route_module.intent_route(state)
    assert result["response_language"] == "English"


def test_intent_route_explicit_request_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式要求优先：本轮明确提出语言要求时，改写会话偏好。"""
    import importlib

    intent_route_module = importlib.import_module("agent.nodes.intent_route")

    monkeypatch.setattr(
        intent_route_module,
        "chat_qwen_json",
        lambda prompt, system=None: '{"category": "简单问答", "rewritten_query": "library hours", '
        '"input_language": "English", "requested_language": "中文"}',
    )
    state = {
        "user_input": "Please reply in Chinese.",
        "conversation_history": [],
        "user_profile": {"role": "留学生"},
        "response_language": "English",
    }
    result = intent_route_module.intent_route(state)
    assert result["response_language"] == "中文"


# ==================== 记忆写入职责纯化 ====================


def test_memory_write_passthrough_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """记忆写入：只写记忆不碰回复内容，final_response 原样透传。"""
    import importlib

    memory_write_module = importlib.import_module("agent.nodes.memory_write")

    monkeypatch.setattr(
        memory_write_module,
        "chat_qwen",
        lambda prompt, system=None, temperature=0.1: "用户问了选课截止时间",
    )
    captured: dict = {}

    def _fake_insert(user_id: str, session_id: str, summary: str, intent: str) -> None:
        captured.update(user_id=user_id, summary=summary, intent=intent)

    monkeypatch.setattr(memory_write_module, "insert_memory", _fake_insert)
    state = {
        "user_input": "选课截止时间是什么时候？",
        "final_response": "选课截止时间为 9 月 15 日。",
        "user_id": "u1",
        "session_id": "s1",
        "intent": IntentResult(
            category="简单问答", confidence=0.9, rewritten_query="选课截止时间"
        ),
    }
    result = memory_write_module.memory_write(state)
    # 职责纯化：不再写 final_response（回复加工由 care_suffix 承担）
    assert "final_response" not in result
    assert result["should_end"] is True
    assert captured["intent"] == "简单问答"
    assert captured["summary"] == "用户问了选课截止时间"
