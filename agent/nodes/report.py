"""节点 2：上报——高危情况触发，记录到独立数据库并通知相关人员。纯逻辑节点。"""
from agent.state import AgentState
from emotion.reporting import build_report_record, submit_report


def report(state: AgentState) -> dict:
    """生成脱敏上报记录并提交，不修改 State。"""
    emotion = state["emotion"]

    record = build_report_record(
        user_id=state["user_id"],
        trigger_text=state["user_input"],
        emotion_level=emotion.level,
        emotion_confidence=emotion.confidence,
        recent_context=state.get("conversation_history", []),
    )
    submit_report(record)

    return {}  # 不修改 State
