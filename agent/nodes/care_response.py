"""节点 3：关怀回复——高危场景优先处理情绪，暂缓回答原始问题。"""
from agent.llm_clients import chat_deepseek
from agent.state import AgentState

# 高危场景的回复模板（模型调用失败时兜底，保证求助信息一定送达）
CARE_TEMPLATE = """我注意到你现在可能正在经历一些很困难的时刻。你的感受很重要，你不需要独自面对。

复旦大学心理咨询中心：
- 预约电话：021-xxxxxxx（24小时）
- 线上预约：xxx.fudan.edu.cn
- 紧急求助：120 / 110

如果你现在处于紧急危险中，请立即拨打120或联系身边信任的人。

你愿意和我多聊聊吗？我会一直在这里。"""

# 关怀回复生成 Prompt：需要高质量的共情表达，用最强模型（DeepSeek-V3）
CARE_PROMPT = """你正在与一位可能处于心理危机中的复旦大学学生对话。
对方刚才表达了令人担忧的情绪。请以温暖、真诚的方式回应，
并提供复旦大学心理咨询中心的联系方式。

要求：
1. 先共情，不要说教
2. 提供具体的求助渠道
3. 表达"你不需要独自面对"
4. 不要使用模板化的套话

复旦大学心理咨询中心电话：021-xxxxxxx（24小时）
紧急求助：120"""


def care_response(state: AgentState) -> dict:
    """生成针对高危情况的关怀回复，暂缓回答原始问题，直接结束流程。"""
    try:
        content = chat_deepseek(CARE_PROMPT, temperature=0.7)  # 共情表达用较高温度
        if not content.strip():
            content = CARE_TEMPLATE
    except Exception:
        content = CARE_TEMPLATE

    return {"final_response": content, "should_end": True}
