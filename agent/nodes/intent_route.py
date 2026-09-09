"""节点：意图路由——意图分类 + Query 改写 + 回复语言判定，三项任务一次调用完成。

合并为一次轻量模型调用（而非三次）的原因：
- 意图分类决定后续走哪条处理分支（检索策略差异巨大）
- Query 改写直接决定检索质量（"怎么选不上课啊"这类口语表述
  与知识库文档语言差距大，不改写会导致召回失败）
- 语言判定决定 generate / care_suffix 的输出语言，且经 State
  跨轮持久化后成为会话级偏好

路由失败兜底：任何异常（网络 / JSON 解析）均回退为"简单问答"
+ 原始 query 直接检索 + 中文回复，保证主流程永不中断。
"""
from agent.llm_clients import chat_qwen_json, parse_json_response
from agent.state import AgentState, IntentResult

# 意图分类 + Query 改写 + 语言判定的一体化指令
INTENT_PROMPT = """你是一个校园问答系统的意图分类器。请分析用户的问题，完成三个任务：

任务1：分类意图（从以下四类中选一个）
- 简单问答：可以从单个文档片段回答的事实性问题（如"选课截止时间是什么时候"）
- 复杂查询：需要综合多个文档片段或推理的问题（如"留学生选课和本科生有什么区别"）
- FAQ：高频常见问题，可能有标准答案（如"校园卡丢了怎么办"）
- 闲聊越界：与校园信息无关的闲聊，或超出系统能力范围的问题（如"今天股票行情"）
注意：调整偏好的指令类输入（如"请用中文回答""以后用英文"）不是闲聊，
请结合对话历史理解为用户的设置请求，归入"简单问答"。

任务2：将用户的问题改写为更适合检索的表述（保留用户语言）
例如：
- "怎么选不上课啊" → "复旦大学本科生选课失败常见原因及解决方法"
- "那个什么截止日期" → "复旦大学本学期选课截止时间"

任务3：判断用户期望的回复语言
- 若用户在当前输入或对话历史中明确要求某语言（如"please reply in English"），
  以该要求为准
- 否则跟随用户当前输入的语言

用户身份：{role}
对话历史（最近3轮）：
{history}
用户问题：{user_input}

请以JSON格式输出：
{{"category": "简单问答", "rewritten_query": "改写后的检索query", "response_language": "中文"}}"""

# 合法意图类别白名单（模型输出不在其中时按简单问答兜底）
VALID_CATEGORIES: tuple = ("简单问答", "复杂查询", "FAQ", "闲聊越界")
# 合法语言白名单（模型输出其他值时按中文兜底）
VALID_LANGUAGES: tuple = ("中文", "English")


def intent_route(state: AgentState) -> dict:
    """调用轻量模型完成意图分类 + Query 改写 + 语言判定，写入 intent / response_language。"""
    user_input = state["user_input"]
    history = state.get("conversation_history", [])
    user_profile = state.get("user_profile", {})
    # 上一轮的语言偏好作为本轮默认值：State 持久化使"说过一次一直生效"成为可能
    previous_language = state.get("response_language", "中文")

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
        response_language = result.get("response_language", previous_language)
    except (ValueError, KeyError, TypeError) as parse_error:
        # 路由失败兜底：按简单问答处理，用原始输入直接检索，沿用既有语言偏好
        print(f"[intent_route] 意图识别失败，按简单问答处理: {parse_error}")
        category, rewritten_query = "简单问答", user_input
        response_language = previous_language

    # 非法类别防御：模型偶发输出白名单外词汇时归入简单问答
    if category not in VALID_CATEGORIES:
        category = "简单问答"
    # 非法语言防御：异常值回落为上一轮偏好
    if response_language not in VALID_LANGUAGES:
        response_language = previous_language

    return {
        "intent": IntentResult(
            category=category,
            confidence=0.9,
            rewritten_query=rewritten_query,
        ),
        "response_language": response_language,
    }


def route_after_intent(state: AgentState) -> str:
    """意图路由后的条件路由：三条处理分支。

    - 简单问答 / FAQ → 统一走 retrieve（内部先做 FAQ 预查，命中直返）
      FAQ 与简单问答的边界本就模糊，检索入口合流后快路径
      不再依赖意图分类的准确性
    - 复杂查询 → 子查询拆解检索
    - 闲聊越界 → 直接兜底回复（不检索不生成，节省资源）
    """
    category = state["intent"].category
    if category in ("简单问答", "FAQ"):
        return "retrieve"
    if category == "复杂查询":
        return "plan_and_retrieve"
    return "chitchat_response"
