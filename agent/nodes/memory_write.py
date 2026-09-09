"""节点：记忆写入——本轮对话摘要写入长期记忆，职责纯化不再触碰回复内容。

职责边界（上游 care_suffix 已产出 final_response）：
- 本节点只负责记忆：生成摘要 + 写库 + 结束标记
- 回复加工（分级关怀后缀等）一律由 care_suffix 节点承担，
  两者通过 State 解耦，各自可独立演化与测试

短期记忆：LangGraph Checkpointer 自动持久化（thread_id 即 session_id），
本节点无需处理。
长期记忆：调用轻量模型生成一句话摘要，写入 user_memory 表，
供后续会话的生成环节做个性化注入。
"""
from agent.llm_clients import chat_qwen
from agent.memory_store import insert_memory
from agent.state import AgentState

SUMMARY_PROMPT = """请用一句话概括以下对话的核心信息，包括用户问了什么、得到了什么答案。
要求：不超过 50 字，去掉所有修饰成分，只保留事实。

用户：{user_input}
助手：{response}"""


def memory_write(state: AgentState) -> dict:
    """生成本轮对话摘要写入长期记忆，透传 final_response 并结束流程。"""
    user_input = state["user_input"]
    response = state["final_response"]
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")
    intent = state.get("intent")

    # 摘要生成：失败时降级为截断拼接（保证记忆链路不因模型故障中断）
    try:
        summary = chat_qwen(SUMMARY_PROMPT.format(user_input=user_input, response=response))
    except Exception as summary_error:
        print(f"[memory_write] 摘要生成失败，降级为截断摘要: {summary_error}")
        summary = f"用户：{user_input[:30]}；助手：{response[:30]}"

    # 写入长期记忆（intent 为 None 的兜底：高危路径不经过本节点，正常不会发生）
    insert_memory(
        user_id=user_id,
        session_id=session_id,
        summary=summary,
        intent=intent.category if intent else "简单问答",
    )

    return {"memory_summary": summary, "should_end": True}
