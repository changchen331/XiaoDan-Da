"""节点：闲聊回复——与校园信息无关或越界的问题，友好引导回校园话题。

输出 generated_response（而非直接 final_response）：闲聊用户同样可能
处于轻度/中度负面情绪，统一经过下游 care_suffix 节点获得分级关怀，
再由其条件路由决定不写长期记忆直接结束。
"""
from agent.llm_clients import chat_deepseek
from agent.state import AgentState

CHITCHAT_SYSTEM = """你是"小旦答"，复旦大学的校园智能问答助手。
用户的问题与校园信息无关，或者超出了你的能力范围。
请友好地引导用户回到校园相关的话题，同时保持亲切的语气。
请使用与用户提问相同的语言回复。
不要编造答案，不要回答与校园无关的问题。"""


def chitchat_response(state: AgentState) -> dict:
    """闲聊 / 越界问题：直接生成兜底回复，不进入检索和生成环节。"""
    content = chat_deepseek(state["user_input"], system=CHITCHAT_SYSTEM)

    return {"generated_response": content}
