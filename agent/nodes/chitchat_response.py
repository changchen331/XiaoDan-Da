"""节点：闲聊回复——与校园信息无关或越界的问题，友好引导回校园话题。

输出 generated_response（而非直接 final_response）：闲聊用户同样可能
处于轻度/中度负面情绪，统一经过下游 care_suffix 节点获得分级关怀，
再由其条件路由决定不写长期记忆直接结束。
"""
from agent.llm_clients import chat_deepseek
from agent.state import AgentState

# 两条安全约束（内部机制不外泄 / 不迎合功利选课）在闲聊与生成两个入口都写：
# 红队用例 off_topic_06 / privacy_05 的路由落点不保证稳定（可能被判为
# "闲聊越界"或"简单问答"），只修一条路径会在重跑时漏网（v2-plan 2.1 #13）。
CHITCHAT_SYSTEM = """你是"小旦答"，复旦大学的校园智能问答助手。
用户的问题与校园信息无关，或者超出了你的能力范围。
请友好地引导用户回到校园相关的话题，同时保持亲切的语气。
请使用与用户提问相同的语言回复。
不要编造答案，不要回答与校园无关的问题。

输出要求（面向聊天窗口）：
- 用自然口语，不使用 Markdown 标题、加粗、编号列表
- 不要自我贬低式推脱（如"我只是校园助手，帮不上忙"）：
  简短说明边界后，立刻给出你可以帮上什么
- 不得自行编造任何联系方式（电话、邮箱、网址）：
  需要给出求助渠道时，只说"建议联系学校心理咨询中心或向辅导员求助"，不要写出具体号码
  （求助渠道由下游节点按情绪等级统一附加，无需在此生成）
- 涉及系统内部实现（是否存储用户信息、如何存储、检索 / 记忆机制）时：
  不描述任何内部细节，只说明无法提供此类信息，并建议向学校相关部门咨询
- 涉及"哪门课 / 哪位老师好混学分、给分宽松、作业压力小"等功利选课请求时：
  不提供此类推荐或暗示，引导以培养方案与课程内容为准"""

# 端点不可用时的兜底引导语（中英双套）
FALLBACK_ZH = "抱歉，我暂时无法回应。你可以问我复旦的选课、培养、宿舍、后勤等校园事务问题。"
FALLBACK_EN = (
    "Sorry, I can't respond right now. "
    "You can ask me about Fudan campus matters such as courses, programs, housing, and logistics."
)


def chitchat_response(state: AgentState) -> dict:
    """闲聊 / 越界问题：直接生成兜底回复，不进入检索和生成环节。"""
    try:
        content = chat_deepseek(state["user_input"], system=CHITCHAT_SYSTEM)
    except Exception as chitchat_error:
        # 兜底：闲聊本就是低价值路径，更不该因端点故障中断整轮
        print(f"[chitchat_response] 生成失败，返回兜底引导语: {chitchat_error}")
        content = (
            FALLBACK_EN if state.get("response_language") == "English" else FALLBACK_ZH
        )

    return {"generated_response": content}
