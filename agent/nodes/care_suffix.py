"""节点：关怀后缀——按情绪分级将 generated_response 加工为 final_response。

设计动机（对应架构文档"分级响应策略"）：
- 正常：原样透传，零额外成本
- 轻度困扰：末尾附加一句自然关怀（不阻断问答，避免"狼来了"效应）
- 中度困扰：关怀 + 心理求助渠道信息（从 .env 注入，LLM 只负责衔接措辞）

职责边界：本节点只做"回复加工"，不写记忆、不路由——
记忆写入由下游 memory_write 独立承担，节点单一职责。

位置在 memory_write 之前的原因：所有正常问答输出路径（FAQ 命中直返 /
质检合格 / 闲聊兜底）都经过本节点，一处实现全覆盖。
高危路径（report → care_response → END）不经过本节点：
高危是整段替换式干预而非附加，且关怀回复面向后台固定中文。
"""

from langgraph.graph import END

from agent.state import AgentState
from config.settings import settings
from infra.llm_clients import chat_qwen

# 轻度困扰：一句自然关怀（LLM 生成，避免固定模板的重复感）
MILD_SUFFIX_PROMPT = """请为下面的校园问答回复追加一句简短的关怀后缀。

要求：
1. 只输出这一句后缀本身，不要重复原回复内容
2. 语气亲切自然，不说教，不超过 30 字
3. 使用{language}输出
4. 后缀单独成段，与原回复之间空一行

原回复：
{response}"""

# 中度困扰：关怀 + 求助渠道（渠道文本原样保留，数字与链接禁止改动）
MODERATE_SUFFIX_PROMPT = """请为下面的校园问答回复追加关怀与求助渠道信息。

要求：
1. 先写一两句共情关怀（语气温暖克制，不说教）
2. 再原样附上"求助渠道"部分：渠道文本必须一字不差地保留，
   禁止改写、省略或编造其中的电话、地点与链接
3. 使用{language}输出
4. 后缀单独成段，与原回复之间空一行

原回复：
{response}

求助渠道（必须原样保留）：
{care_info}"""

# 降级模板：LLM 生成失败时兜底（中英两套，按用户语言选择）
_FALLBACK_SUFFIX_ZH = "听起来你最近可能有些压力，注意休息、照顾好自己。\n\n{care_info}"
_FALLBACK_SUFFIX_EN = (
    "Sounds like things may feel heavy lately. " "Take care of yourself.\n\n{care_info}"
)


def _fallback_suffix(level: str, response_language: str) -> str:
    """生成降级关怀后缀（LLM 不可用时的兜底，确定性输出）。

    :param level: 情绪等级（轻度困扰 / 中度困扰）
    :param response_language: 回复语言（中文 / English）
    :return: 固定模板后缀文本
    """
    if level == "轻度困扰":
        if response_language == "English":
            return (
                "Sounds like things may feel a bit heavy lately. Take care of yourself."
            )
        return "听起来你最近可能有些压力，注意休息、照顾好自己。"

    # 中度困扰：关怀 + 渠道信息（渠道语言跟随配置而非用户偏好，
    # 避免中文渠道信息被强行翻译；若用户为英文则附英文版渠道）
    care_info = (
        settings.CARE_CENTER_INFO_ZH
        if response_language != "English"
        else settings.CARE_CENTER_INFO_EN
    )
    template = (
        _FALLBACK_SUFFIX_ZH if response_language != "English" else _FALLBACK_SUFFIX_EN
    )
    return template.format(care_info=care_info)


def care_suffix(state: AgentState) -> dict:
    """按情绪分级为回复附加关怀内容，产出 final_response 并追加本轮对话历史。

    - 正常 / 无情绪结果：透传 generated_response
    - 轻度 / 中度困扰：LLM 生成自然后缀，失败时降级固定模板

    本节点是非高危路径的**统一收尾处**（正常问答、FAQ 快路径、闲聊都汇聚于此），
    因此把「追加 conversation_history」放在这里，一次覆盖全部出口。
    """
    emotion = state.get("emotion")
    response = state["generated_response"]
    response_language = state.get("response_language", "中文")

    suffix = ""
    # 正常（或情绪缺失的异常兜底）：不附加任何内容，避免打扰
    if emotion is not None and emotion.level not in ("正常", "高危"):
        level = emotion.level
        language_name = "英文" if response_language == "English" else "中文"
        try:
            if level == "轻度困扰":
                suffix = chat_qwen(
                    MILD_SUFFIX_PROMPT.format(language=language_name, response=response)
                )
            else:  # 中度困扰
                care_info = (
                    settings.CARE_CENTER_INFO_ZH
                    if response_language != "English"
                    else settings.CARE_CENTER_INFO_EN
                )
                suffix = chat_qwen(
                    MODERATE_SUFFIX_PROMPT.format(
                        language=language_name, response=response, care_info=care_info
                    )
                )
        except Exception as suffix_error:
            # 降级链：LLM 后缀生成失败不阻断回复输出，落回固定模板
            print(f"[care_suffix] 后缀生成失败，降级固定模板: {suffix_error}")
            suffix = _fallback_suffix(level, response_language)

    final_response = f"{response}\n\n{suffix}" if suffix else response

    # 追加本轮问答到短期记忆：该字段靠 Checkpointer 跨轮保留（invoke 不会重置它），
    # 供下一轮的规则引擎（多轮累积升级）与生成节点（上下文）读取。
    # 上限取 HISTORY_MAX_ROUNDS 轮（每轮 user + assistant 共两条），防止状态无限膨胀。
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["user_input"]})
    history.append({"role": "assistant", "content": final_response})

    return {
        "final_response": final_response,
        "conversation_history": history[-settings.HISTORY_MAX_ROUNDS * 2 :],
    }


def route_after_care_suffix(state: AgentState) -> str:
    """关怀后缀后的条件路由：闲聊越界不写长期记忆，其余写入。

    闲聊无事实内容、摘要价值低且会稀释 user_memory 的个性化信号，
    因此直接结束；意图字段缺失的异常情形同样按不写记忆处理。
    """
    intent = state.get("intent")
    if intent is None or intent.category == "闲聊越界":
        return END
    return "memory_write"
