"""节点：意图路由——判断用户意图 + 将口语化 query 改写为检索友好的表述。

两项任务合并为一次轻量模型调用（而非两次）：
- 意图分类决定后续走哪条处理分支（检索策略差异巨大）
- Query 改写直接决定检索质量（"怎么选不上课啊"这类口语表述
  与知识库文档语言差距大，不改写会导致召回失败）

路由失败兜底：任何异常（网络 / JSON 解析）均回退为"简单问答"
+ 原始 query 直接检索，保证主流程永不中断。
"""
from agent.llm_clients import chat_qwen_json, parse_json_response
from agent.state import AgentState, IntentResult

# 意图分类 + Query 改写的一体化指令
INTENT_PROMPT = """你是一个校园问答系统的意图分类器。请分析用户的问题，完成两个任务：

任务1：分类意图（从以下四类中选一个）
- 简单问答：可以从单个文档片段回答的事实性问题（如"选课截止时间是什么时候"）
- 复杂查询：需要综合多个文档片段或推理的问题（如"留学生选课和本科生有什么区别"）
- FAQ：高频常见问题，可能有标准答案（如"校园卡丢了怎么办"）
- 闲聊越界：与校园信息无关的闲聊，或超出系统能力范围的问题（如"今天股票行情"）

任务2：将用户的问题改写为更适合检索的表述（保留用户语言）
例如：
- "怎么选不上课啊" → "复旦大学本科生选课失败常见原因及解决方法"
- "那个什么截止日期" → "复旦大学本学期选课截止时间"

用户身份：{role}
对话历史（最近3轮）：
{history}
用户问题：{user_input}

请以JSON格式输出：
{{"category": "简单问答", "rewritten_query": "改写后的检索query"}}"""

# 合法意图类别白名单（模型输出不在其中时按简单问答兜底）
VALID_CATEGORIES: tuple = ("简单问答", "复杂查询", "FAQ", "闲聊越界")


def intent_route(state: AgentState) -> dict:
    """调用轻量模型完成意图分类 + Query 改写，写入 intent 字段。"""
    user_input = state["user_input"]
    history = state.get("conversation_history", [])
    user_profile = state.get("user_profile", {})

    prompt = INTENT_PROMPT.format(
        role=user_profile.get("role", "未知"),
        history=history[-3:] if history else "无",
        user_input=user_input,
    )

    try:
        raw = chat_qwen_json(prompt)
        result = parse_json_response(raw)
        category = result["category"]
        rewritten_query = result["rewritten_query"]
    except (ValueError, KeyError, TypeError) as parse_error:
        # 路由失败兜底：按简单问答处理，用原始输入直接检索
        print(f"[intent_route] 意图识别失败，按简单问答处理: {parse_error}")
        category, rewritten_query = "简单问答", user_input

    # 非法类别防御：模型偶发输出白名单外词汇时归入简单问答
    if category not in VALID_CATEGORIES:
        category = "简单问答"

    return {"intent": IntentResult(
        category=category,
        confidence=0.9,
        rewritten_query=rewritten_query,
    )}


def route_after_intent(state: AgentState) -> str:
    """意图路由后的条件路由：四条处理分支。

    - 简单问答 → 单次混合检索
    - 复杂查询 → 子查询拆解检索
    - FAQ      → FAQ 专用索引优先
    - 闲聊越界 → 直接兜底回复（不检索不生成，节省资源）
    """
    category = state["intent"].category
    if category == "简单问答":
        return "retrieve"
    if category == "复杂查询":
        return "plan_and_retrieve"
    if category == "FAQ":
        return "faq_match"
    return "chitchat_response"
