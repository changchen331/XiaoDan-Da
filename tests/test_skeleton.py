"""核心逻辑单元测试：不依赖任何外部服务（LLM / 数据库 / 向量库），
覆盖规则引擎、切分策略、文本清洗、两层降级、学期计算与 JSON 解析。

运行：python -m pytest tests/test_skeleton.py -v
"""
from datetime import date

from agent.llm_clients import parse_json_response
from config.settings import get_current_semester
from emotion.detector import detect_emotion
from emotion.rule_engine import RuleEngine
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
    """严重自我否定：规则层定级中度（留给融合逻辑二次确认）。"""
    engine = RuleEngine()
    assert engine.check("我真是个废物")["level"] == "中度"


def test_rule_engine_multi_round_escalation() -> None:
    """多轮累积：连续 3 轮负面弱信号 → 升级中度。"""
    engine = RuleEngine()
    history = [
        {"role": "user", "content": "考试好难啊"},
        {"role": "assistant", "content": "加油"},
        {"role": "user", "content": "感觉压力好大"},
        {"role": "assistant", "content": "注意休息"},
        {"role": "user", "content": "最近好累"},
    ]
    assert engine.check("今天也好累", history)["level"] == "中度"


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
    text = ("关于2026年选课安排的通知\n第一条内容。\n\n"
            "关于宿舍调整的通知\n第二条内容。")
    chunks = chunk_by_type(text, "通知")
    assert len(chunks) == 2


def test_chunker_manual_splits_long_text() -> None:
    """手册切分：长文本被递归切为多个 chunk 且带重叠。"""
    long_text = "\n\n".join(f"第{i}段落内容。" + "课程设置的详细说明。" * 30
                          for i in range(20))
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

    expected = {"user_input", "user_id", "session_id", "emotion", "intent",
                "retrieved_contexts", "faq_hit", "generated_response", "quality",
                "retry_count", "memory_summary", "final_response", "should_end"}
    assert expected.issubset(set(AgentState.__annotations__))
