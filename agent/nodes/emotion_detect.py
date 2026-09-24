"""节点 1：情绪检测——每轮输入必经，先于一切问答逻辑。"""
from agent.state import AgentState, EmotionResult
from emotion.detector import detect_emotion


def emotion_detect(state: AgentState) -> dict:
    """对用户输入做情绪分析，写入 emotion 字段。"""
    text = state["user_input"]
    history = state.get("conversation_history", [])

    result = detect_emotion(text, history)

    return {"emotion": EmotionResult(
        level=result["level"],
        source=result["source"],
        confidence=result["confidence"],
        referent=result.get("referent"),
    )}


def route_after_emotion(state: AgentState) -> str:
    """情绪检测后的条件路由：高危走上报分支，其余走正常问答流程。"""
    emotion = state.get("emotion")
    if emotion is not None and emotion.level == "高危":
        return "report"
    return "intent_route"
