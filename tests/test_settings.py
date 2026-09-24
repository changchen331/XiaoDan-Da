"""配置中心与状态契约：学期计算、JSON 容错解析、AgentState 字段定义。"""

from datetime import date

from config.settings import get_current_semester
from infra.llm_clients import parse_json_response

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
