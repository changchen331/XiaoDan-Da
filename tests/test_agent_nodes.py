"""Agent 节点与条件路由：意图路由与语言偏好、FAQ 轻量校验、关怀后缀分级、路由分支与记忆写入职责。"""

import pytest

from agent.state import EmotionResult, IntentResult
from config.settings import settings

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
    # 断言「渠道文本来自配置」而不是断言某几个字面词：
    # 渠道内容属于部署配置（.env），换一所学校、换一个电话就会让字面断言失效——
    # 那样的用例测的是配置内容，不是代码行为
    assert settings.CARE_CENTER_INFO_ZH in result["final_response"]


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


def test_intent_route_explicit_request_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


# ==================== FAQ 快路径：匹配与校验的输入口径 ====================


class _StubFaqIndex:
    """FAQ 索引桩：按 query 返回预设相似度，避免测试连 Milvus。"""

    def __init__(self, scores: dict) -> None:
        self.scores = scores

    def match(self, query: str) -> dict | None:
        if query not in self.scores:
            return None
        return {
            "question": f"标准问题（{query}）",
            "answer": "标准答案正文，长度足够通过校验。",
            "score": self.scores[query],
        }


def test_faq_precheck_verifies_with_user_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAQ 校验必须用**用户原话**，而不是改写后的 query。

    实测缺陷：意图路由把"复旦大学图书馆学位论文提交系统是什么？"改写成
    "…介绍及使用指南"，改写把问法放宽了，校验器于是判标准答案"答不全"
    （answer_match=false），相似度 0.917 的高置信命中被误拒、快路径整条失效。
    """
    import importlib

    retrieve_module = importlib.import_module("agent.nodes.retrieve")

    monkeypatch.setattr(
        "knowledge_base.faq.get_faq_index",
        lambda: _StubFaqIndex({"用户原话问句": 0.95}),
    )
    captured: dict = {}

    def _fake_verify(query: str, question: str, answer: str, language: str) -> bool:
        captured["query"] = query
        return True

    monkeypatch.setattr(retrieve_module, "verify_faq_match", _fake_verify)
    matched = retrieve_module._faq_precheck("改写后的问句", "中文", "用户原话问句")

    assert matched is not None
    assert captured["query"] == "用户原话问句"


def test_faq_precheck_takes_higher_score_of_both_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原话与改写 query **都参与匹配、取更高分**。

    实测：同一个问题用原话命中 0.953、用改写只有 0.855——后者贴着 0.85 阈值，
    命中与否随改写措辞抖动。只用一个会让快路径时灵时不灵。
    """
    import importlib

    retrieve_module = importlib.import_module("agent.nodes.retrieve")

    monkeypatch.setattr(
        "knowledge_base.faq.get_faq_index",
        lambda: _StubFaqIndex({"改写问句": 0.86, "原话问句": 0.95}),
    )
    monkeypatch.setattr(retrieve_module, "verify_faq_match", lambda *a, **k: True)

    matched = retrieve_module._faq_precheck("改写问句", "中文", "原话问句")
    assert matched is not None
    assert matched["score"] == 0.95


# ==================== 红队失败项回归（隐私 / 功利选课） ====================


def test_generation_prompts_cover_red_team_failures() -> None:
    """两个生成入口的 Prompt 必须携带两条红队约束。

    privacy_05（不得暴露内部机制 / 不描述存储细节）与
    off_topic_06（不迎合功利选课）的修复落在 Prompt 文本上——
    用例的路由落点不保证稳定，因此闲聊与生成两个入口都要覆盖。
    """
    from agent.nodes.chitchat_response import CHITCHAT_SYSTEM
    from agent.nodes.generate import GENERATE_PROMPT

    # generate：不得暴露内部机制（含不得复述"用户背景"条目）+ 不迎合功利选课
    assert "内部机制" in GENERATE_PROMPT
    assert "来源话题" in GENERATE_PROMPT
    assert "功利选课" in GENERATE_PROMPT

    # chitchat：同一用例可能被判为"闲聊越界"而走此分支，同样要有约束
    assert "内部实现" in CHITCHAT_SYSTEM
    assert "功利选课" in CHITCHAT_SYSTEM


# ==================== 质检重试放宽检索 ====================


def test_retrieve_relaxes_retrieval_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试时的检索输入必须与上一次不同（否则重试没有信息增益）。

    验证三处放宽：去人群过滤（保留学期时效）、候选池翻倍、重排片段 +2。
    用 sys.modules 注入假检索模块：真实模块会连带导入 FlagEmbedding
    （实测约 11s），单测不应为参数透传付这个加载代价。
    """
    import importlib
    import sys
    import types

    retrieve_module = importlib.import_module("agent.nodes.retrieve")

    captured: dict = {}

    class _FakeService:
        def hybrid_search(
            self, query: str, filter_expr: str = "", top_k: int | None = None
        ) -> list:
            captured["filter_expr"] = filter_expr
            captured["top_k"] = top_k
            return []

        def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
            captured["rerank_top_k"] = top_k
            return []

    fake_retrieval = types.ModuleType("knowledge_base.retrieval")
    fake_retrieval.get_retrieval_service = lambda: _FakeService()
    monkeypatch.setitem(sys.modules, "knowledge_base.retrieval", fake_retrieval)
    monkeypatch.setattr(
        retrieve_module, "_faq_precheck", lambda query, language, user_input=None: None
    )

    state = {
        "user_input": "选课截止时间",
        "intent": IntentResult(
            category="简单问答", confidence=0.9, rewritten_query="选课截止时间"
        ),
        "user_profile": {"role": "本科生"},
        "retry_count": 0,
    }

    # 首次检索：严格过滤 + 基准召回数
    retrieve_module.retrieve(state)
    assert "target_audience" in captured["filter_expr"]
    assert captured["top_k"] == settings.HYBRID_TOP_K
    assert captured["rerank_top_k"] == settings.RERANK_TOP_K

    # 质检重试第 1 次：去人群过滤（学期时效保留）+ 候选翻倍 + 重排 +2
    state["retry_count"] = 1
    retrieve_module.retrieve(state)
    assert "target_audience" not in captured["filter_expr"]
    assert "valid_semester" in captured["filter_expr"]
    assert captured["top_k"] == settings.HYBRID_TOP_K * 2
    assert captured["rerank_top_k"] == settings.RERANK_TOP_K + 2

    # 重试第 2 次必须继续放宽：与上一次参数相同 = 第 2 次重试仍无信息增益
    state["retry_count"] = 2
    retrieve_module.retrieve(state)
    assert captured["top_k"] == settings.HYBRID_TOP_K * 3
    assert captured["rerank_top_k"] == settings.RERANK_TOP_K + 4


# ==================== 上报：危险表达的指向标注 ====================


def _alert_state(referent: str | None) -> dict:
    """构造上报节点所需的 State（高危 + 指定指向）。"""
    return {
        "user_input": "我室友说他活不下去了",
        "user_id": "u1",
        "session_id": "s1",
        "conversation_history": [],
        "emotion": EmotionResult(
            level="高危", source="规则引擎", confidence=1.0, referent=referent
        ),
    }


def test_report_keeps_detector_referent_without_extra_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检测侧已给出指向时直接入库：不再补标注，也就不产生额外 LLM 调用。"""
    import importlib

    report_module = importlib.import_module("agent.nodes.report")
    captured: dict = {}
    spawned: list = []

    def _fake_submit(record: dict) -> int:
        captured.update(record)
        return 1

    monkeypatch.setattr(report_module, "submit_report", _fake_submit)
    monkeypatch.setattr(
        report_module, "_annotate_referent_async", lambda *args: spawned.append(args)
    )

    assert report_module.report(_alert_state(referent="他人")) == {}
    assert captured["referent"] == "他人"
    assert spawned == []


def test_report_spawns_annotation_for_rule_high_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """规则命中的高危：指向未知 → 提交后交由**后台**补标注（不占用用户路径）。"""
    import importlib

    report_module = importlib.import_module("agent.nodes.report")
    captured: dict = {}
    spawned: list = []

    def _fake_submit(record: dict) -> int:
        captured.update(record)
        return 7

    monkeypatch.setattr(report_module, "submit_report", _fake_submit)
    monkeypatch.setattr(
        report_module, "_annotate_referent_async", lambda *args: spawned.append(args)
    )

    report_module.report(_alert_state(referent=None))
    assert captured["referent"] is None
    assert spawned == [(7, "我室友说他活不下去了")]


def test_report_skips_annotation_when_write_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写库失败（无记录 id）时不必再补标注：没有可回写的行。"""
    import importlib

    report_module = importlib.import_module("agent.nodes.report")
    spawned: list = []
    monkeypatch.setattr(report_module, "submit_report", lambda record: None)
    monkeypatch.setattr(
        report_module, "_annotate_referent_async", lambda *args: spawned.append(args)
    )

    report_module.report(_alert_state(referent=None))
    assert spawned == []


def test_annotation_writes_referent_and_swallows_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """补标注：把指向写回记录；语义层故障时只打日志，不影响上报本身。"""
    import importlib

    import agent.llm_clients as llm

    report_module = importlib.import_module("agent.nodes.report")
    written: list = []

    monkeypatch.setattr(settings, "EMOTION_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(report_module, "judge_referent", lambda text: "他人")
    monkeypatch.setattr(
        report_module,
        "annotate_alert_referent",
        lambda alert_id, referent: written.append((alert_id, referent)),
    )

    report_module._annotate_referent(7, "我室友说他活不下去了")
    assert written == [(7, "他人")]

    def _outage(text: str) -> str:
        raise llm.LLMUnavailableError("三跳均失败")

    monkeypatch.setattr(report_module, "judge_referent", _outage)
    report_module._annotate_referent(7, "我室友说他活不下去了")  # 不得抛出
    assert written == [(7, "他人")]

    # 开关关闭：不在安全路径上引入外部调用
    monkeypatch.setattr(settings, "EMOTION_SEMANTIC_ENABLED", False)
    monkeypatch.setattr(report_module, "judge_referent", lambda text: "本人")
    report_module._annotate_referent(7, "我室友说他活不下去了")
    assert written == [(7, "他人")]
