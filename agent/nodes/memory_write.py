"""节点：记忆写入——本轮对话摘要写入长期记忆，同时输出最终回答。

职责拆分：
- 短期记忆：LangGraph Checkpointer 自动持久化（thread_id 即 session_id），
  本节点无需处理
- 长期记忆：调用轻量模型生成一句话摘要，写入 user_memory 表，
  供后续会话的生成环节做个性化注入

FAQ 命中的路径也经过本节点（标准答案同样值得记忆——
用户连续问同一 FAQ 时可观察出高频问题，为 FAQ 库扩充提供依据）。
"""
from agent.llm_clients import chat_qwen
from agent.memory_store import insert_memory
from agent.state import AgentState

SUMMARY_PROMPT = """请用一句话概括以下对话的核心信息，包括用户问了什么、得到了什么答案。
要求：不超过 50 字，去掉所有修饰成分，只保留事实。

用户：{user_input}
助手：{response}"""


def memory_write(state: AgentState) -> dict:
    """生成本轮对话摘要写入长期记忆，输出 final_response 并结束流程。"""
    user_input = state["user_input"]
    response = state["generated_response"]
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

    return {"memory_summary": summary, "final_response": response, "should_end": True}
