"""节点 3：关怀回复——高危场景优先处理情绪，暂缓回答原始问题。"""
from agent.llm_clients import chat_deepseek
from agent.state import AgentState
from config.settings import settings

# 高危场景的回复模板（模型调用失败时兜底，保证求助信息一定送达）
# 渠道信息从配置注入而非硬编码：与 care_suffix 保持同一原则——
# 仓库内只留占位符（公开仓库不携带真实联系方式），且禁止模型编造电话
CARE_TEMPLATE = """我注意到你现在可能正在经历一些很困难的时刻。你的感受很重要，你不需要独自面对。

{care_info}

紧急求助：120 / 110

如果你现在处于紧急危险中，请立即拨打120或联系身边信任的人。

你愿意和我多聊聊吗？我会一直在这里。"""

# 关怀回复生成 Prompt：需要高质量的共情表达，用最强模型（DeepSeek-V3）
CARE_PROMPT = """你正在与一位可能处于心理危机中的复旦大学学生对话。
对方刚才表达了令人担忧的情绪。请以温暖、真诚的方式回应。

要求：
1. 先共情，不要说教
2. 提供具体的求助渠道：必须原样保留下方给出的渠道信息，
   禁止改写、省略或编造其中的电话、地点与链接
3. 表达"你不需要独自面对"
4. 不要使用模板化的套话

可提供的求助渠道（必须原样保留）：
{care_info}

紧急求助：120"""


def care_response(state: AgentState) -> dict:
    """生成针对高危情况的关怀回复，暂缓回答原始问题，直接结束流程。

    高危关怀固定使用中文（面向后台处理者与校内求助场景），
    因此取中文渠道配置，不读用户的 response_language。
    """
    care_info = settings.CARE_CENTER_INFO_ZH
    try:
        content = chat_deepseek(CARE_PROMPT.format(care_info=care_info), temperature=0.7)
        if not content.strip():
            content = CARE_TEMPLATE.format(care_info=care_info)
    except Exception:
        content = CARE_TEMPLATE.format(care_info=care_info)

    # 高危分支不经 care_suffix 直达 END，故此处单独追加对话历史，
    # 与 care_suffix 保持同一口径（下一轮可以参考上一轮说过什么）
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["user_input"]})
    history.append({"role": "assistant", "content": content})

    return {
        "final_response": content,
        "should_end": True,
        "conversation_history": history[-settings.HISTORY_MAX_ROUNDS * 2:],
    }
