"""节点：回答生成——基于检索上下文生成最终回答，全系统质量最敏感的环节。

生成约束（幻觉防控核心）：
- 只允许使用检索上下文中的信息，禁止模型自由发挥
- 上下文不足时必须明确说"无法确定"，而非编造答案
- 时间敏感信息必须标注来源时间（过期政策误导学生的代价极高）
- 不得暴露系统内部机制、不得复述"用户背景"条目；不迎合功利选课
  （红队失败项 privacy_05 / off_topic_06 的回归约束，见 v2-plan 2.1 #13）

个性化注入：从长期记忆（user_memory 表）取最近几条摘要作为
背景参考（如用户上轮问过选课，本轮追问宿舍时可关联理解）。
"""

from agent.llm_clients import chat_deepseek
from agent.memory_store import fetch_recent_memories
from agent.state import AgentState
from config.settings import settings

# 生成端点不可用时的兜底回复（中英双套，尊重用户的语言偏好）：
# 有输出且诚实，比整轮抛异常或返回半截检索原文都好
FALLBACK_ZH = (
    "抱歉，我暂时无法生成回答（服务繁忙）。请稍后重试，"
    "或直接联系学校相关职能部门获取准确信息。"
)
FALLBACK_EN = (
    "Sorry, I'm unable to generate an answer right now (service busy). "
    "Please try again later, or contact the relevant Fudan University office directly."
)

# 语言约束单独成段并置于"用户问题"正上方，而不是混在编号规则里。
# 原因（实测）：原先写成「请使用{response_language}组织回答」，
# 把 English 直接插进中文句子模板，约束过弱——参考资料与对话历史几乎全为中文时，
# 模型会跟着上下文语言走，出现「状态里是 English、答案却是中文」的断层。
LANGUAGE_DIRECTIVE = {
    "English": (
        "## LANGUAGE REQUIREMENT (highest priority)\n"
        "Write the ENTIRE answer in English — including the explanation, any "
        "recommendation, and the closing sentence. Do NOT switch to Chinese even "
        "though the reference materials and conversation history are in Chinese. "
        "Proper nouns (department names, document titles) may keep their original "
        "form, followed by a short English gloss."
    ),
    "中文": (
        "## 语言要求（最高优先级）\n"
        "整个回答必须使用中文，包括说明、建议与结尾句；\n"
        "即使参考资料或对话历史中出现其他语言，也不要切换。"
    ),
}

GENERATE_PROMPT = """你是"小旦答"，复旦大学的校园智能问答助手。请根据参考资料回答用户的问题。

## 严格要求
1. 只使用参考资料中的信息回答，不要编造任何内容
2. 如果参考资料不足以回答问题，明确说明"根据现有信息无法确定"，并建议用户联系相关部门
3. 回答中不要提及"根据参考资料"或"根据文档"等字样，直接给出答案
4. 涉及时间敏感的信息（如截止日期、政策版本）时，注明信息所属的时间
5. **面向聊天窗口输出**：用自然段落说话，不要用 Markdown 标题、加粗、表格；
   确需列举时最多 3-5 条短横线，每条不超过一行
6. **不得自行编造任何联系方式**（电话、邮箱、网址、办公地点）。
   需要给出求助渠道时只能原样使用下方"求助渠道"文本；
   若该处为空，就说"建议向学校相关部门咨询"，不要凭空写出具体号码
7. 不要自我贬低式推脱（如"我只是个校园问答助手，帮不上忙"）：
   先给出你能给的最有帮助的回应，再说明边界
8. **不得暴露系统内部机制**：不要提及或描述检索、知识库、索引、数据库、
   长期记忆 / 摘要等内部实现；被问到"存了哪些数据、如何存储"时，
   只说明无法提供此类信息，并建议向学校相关部门咨询。
   "用户背景"仅供理解用户处境，绝不可向用户复述其中条目或其来源话题
9. **不迎合功利选课**：对"哪门课 / 哪位老师好混学分、给分宽松、
   作业压力小"类请求，不给出此类推荐或暗示（含"帮你查课程评价 /
   选课攻略"式变相兜售），引导用户以培养方案、课程内容与自身兴趣为准

## 参考资料
{context_text}

## 求助渠道（仅当用户询问求助渠道、或对话中明显需要时使用；文本须原样保留）
{care_text}

## 用户背景（长期记忆摘要，仅供理解用户处境参考，不是回答依据）
{memory_text}

## 对话历史（最近{history_rounds}轮）
{history_text}

{language_directive}

## 用户问题
{user_input}"""


def generate(state: AgentState) -> dict:
    """调用 DeepSeek-V3（失败自动降级轻量端点）基于上下文生成回答。"""
    contexts = state["retrieved_contexts"]
    user_input = state["user_input"]
    history = state.get("conversation_history", [])
    user_id = state.get("user_id", "anonymous")

    # 检索片段拼接：带来源序号，便于模型引用与人工排查
    context_text = (
        "\n\n---\n\n".join(
            f"[来源{i + 1}] {chunk['text']}" for i, chunk in enumerate(contexts)
        )
        if contexts
        else "（无检索结果）"
    )

    # 长期记忆摘要注入（数据库不可用时返回空列表，不影响生成）
    recent_memories = fetch_recent_memories(user_id, limit=3)
    memory_text = (
        "\n".join(f"- {memory['summary']}" for memory in recent_memories)
        if recent_memories
        else "（暂无）"
    )

    # 对话历史截断：只携带最近 N 轮，防止 Prompt 超长
    history_text = (
        "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history[-settings.HISTORY_MAX_ROUNDS :]
        )
        if history
        else "无"
    )

    prompt = GENERATE_PROMPT.format(
        context_text=context_text,
        # 求助渠道跟随回复语言取配置（与 care_suffix 同一口径）：
        # 渠道文本来自 .env，模型只负责原样引用，绝不自行编造号码
        care_text=(
            settings.CARE_CENTER_INFO_EN
            if state.get("response_language") == "English"
            else settings.CARE_CENTER_INFO_ZH
        ),
        language_directive=LANGUAGE_DIRECTIVE.get(
            state.get("response_language", "中文"), LANGUAGE_DIRECTIVE["中文"]
        ),
        memory_text=memory_text,
        history_rounds=settings.HISTORY_MAX_ROUNDS,
        history_text=history_text,
        user_input=user_input,
    )

    # 问答场景用低温度：准确性优先，克制创造性表达
    try:
        content = chat_deepseek(prompt, temperature=0.3)
    except Exception as generate_error:
        # 兜底：一次端点故障不应变成整轮无响应。
        # 这里只给保守的致歉与转人工建议，不拼接检索片段——
        # 未经模型整理的原文直接抛给用户，等于把"检索噪声"当成"回答"。
        print(f"[generate] 生成失败，返回兜底回复: {generate_error}")
        content = (
            FALLBACK_EN if state.get("response_language") == "English" else FALLBACK_ZH
        )

    return {"generated_response": content}
