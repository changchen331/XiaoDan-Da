"""节点：回答生成——基于检索上下文生成最终回答，全系统质量最敏感的环节。

生成约束（幻觉防控核心）：
- 只允许使用检索上下文中的信息，禁止模型自由发挥
- 上下文不足时必须明确说"无法确定"，而非编造答案
- 时间敏感信息必须标注来源时间（过期政策误导学生的代价极高）

个性化注入：从长期记忆（user_memory 表）取最近几条摘要作为
背景参考（如用户上轮问过选课，本轮追问宿舍时可关联理解）。
"""
from agent.llm_clients import chat_deepseek
from agent.memory_store import fetch_recent_memories
from agent.state import AgentState
from config.settings import settings

GENERATE_PROMPT = """你是"小旦答"，复旦大学的校园智能问答助手。请根据参考资料回答用户的问题。

## 严格要求
1. 只使用参考资料中的信息回答，不要编造任何内容
2. 如果参考资料不足以回答问题，明确说明"根据现有信息无法确定"，并建议用户联系相关部门
3. 回答中不要提及"根据参考资料"或"根据文档"等字样，直接给出答案
4. 涉及时间敏感的信息（如截止日期、政策版本）时，注明信息所属的时间

## 参考资料
{context_text}

## 用户背景（长期记忆摘要，仅供理解用户处境参考，不是回答依据）
{memory_text}

## 对话历史（最近{history_rounds}轮）
{history_text}

## 用户问题
{user_input}"""


def generate(state: AgentState) -> dict:
    """调用 DeepSeek-V3（失败自动降级轻量端点）基于上下文生成回答。"""
    contexts = state["retrieved_contexts"]
    user_input = state["user_input"]
    history = state.get("conversation_history", [])
    user_id = state.get("user_id", "anonymous")

    # 检索片段拼接：带来源序号，便于模型引用与人工排查
    context_text = "\n\n---\n\n".join(
        f"[来源{i + 1}] {chunk['text']}" for i, chunk in enumerate(contexts)
    ) if contexts else "（无检索结果）"

    # 长期记忆摘要注入（数据库不可用时返回空列表，不影响生成）
    recent_memories = fetch_recent_memories(user_id, limit=3)
    memory_text = "\n".join(
        f"- {memory['summary']}" for memory in recent_memories
    ) if recent_memories else "（暂无）"

    # 对话历史截断：只携带最近 N 轮，防止 Prompt 超长
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-settings.HISTORY_MAX_ROUNDS:]
    ) if history else "无"

    prompt = GENERATE_PROMPT.format(
        context_text=context_text,
        memory_text=memory_text,
        history_rounds=settings.HISTORY_MAX_ROUNDS,
        history_text=history_text,
        user_input=user_input,
    )

    # 问答场景用低温度：准确性优先，克制创造性表达
    content = chat_deepseek(prompt, temperature=0.3)
    return {"generated_response": content}
