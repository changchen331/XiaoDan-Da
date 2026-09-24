"""节点 2：上报——高危情况触发，记录到独立数据库并通知相关人员。纯逻辑节点。"""
import threading

from agent.state import AgentState
from config.settings import settings
from emotion.reporting import (
    annotate_alert_referent,
    build_report_record,
    submit_report,
)
from emotion.semantic import judge_referent
from infra.llm_clients import JSON_TASK_ERRORS


def report(state: AgentState) -> dict:
    """生成脱敏上报记录并提交，不修改 State。"""
    emotion = state["emotion"]
    user_input = state["user_input"]

    record = build_report_record(
        user_id=state["user_id"],
        trigger_text=user_input,
        emotion_level=emotion.level,
        emotion_confidence=emotion.confidence,
        recent_context=state.get("conversation_history", []),
        referent=emotion.referent,
    )
    alert_id = submit_report(record)

    # 规则命中高危时 referent 恒为 None（那条路径为保零延迟不调 LLM），
    # 于是"我室友说他活不下去了"会与本人危机混在同一条记录里。
    # 补标注刻意放到**后台**：记录已落库、邮件已发出，用户侧不该为一次标注
    # 再等一个超时周期（云端全挂时那条链最坏要走三跳）
    if alert_id is not None and emotion.referent is None:
        _annotate_referent_async(alert_id, user_input)

    return {}  # 不修改 State


def _annotate_referent_async(alert_id: int, text: str) -> None:
    """后台补标注（守护线程）：只更新上报记录的指向字段，不改变用户侧响应。"""
    thread = threading.Thread(
        target=_annotate_referent, args=(alert_id, text), daemon=True
    )
    thread.start()


def _annotate_referent(alert_id: int, text: str) -> None:
    """取一次"危险表达指向谁"并回写上报记录；任何失败都只打日志。

    与语义判别层共用开关：部署方关掉了 LLM 语义层时，
    就不该在安全路径上再引入一次外部调用。
    """
    if not settings.EMOTION_SEMANTIC_ENABLED:
        return
    try:
        annotate_alert_referent(alert_id, judge_referent(text))
    except JSON_TASK_ERRORS as referent_error:
        print(f"[report] 指向标注失败，记录维持未标注: {referent_error}")
