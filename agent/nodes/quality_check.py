"""节点：质量评估——不合格则回退检索重试（最多 2 次）。

评估两维度（与 RAGAS 离线指标对齐，构成在线 / 离线一体：
- 忠实度：回答是否完全基于检索上下文，有无编造（幻觉在线拦截）
- 相关性：回答是否切题，有无答非所问

防死循环设计：重试计数在评估节点累加，达到上限（默认 2 次）后
无论质量如何直接放行——第 3 次生成仍不合格时，
一条"不够完美但可用"的回答远优于无限循环或空响应。

重试不会重复同样的检索：retrieve 节点按 retry_count 逐级放宽
候选池与过滤条件（见该节点 docstring），让下一次生成有新的上下文
可用，而不是把同一份结果再生成一遍。
"""
from agent.state import AgentState, QualityResult
from config.settings import settings
from infra.llm_clients import JSON_TASK_ERRORS, chat_qwen_json_parsed

QUALITY_PROMPT = """请评估以下校园问答的回答质量，检查两个维度：

1. 忠实度：回答中的每一条信息是否都能在参考资料中找到依据？有没有编造的内容？
2. 相关性：回答是否直接回应了用户的问题？有没有答非所问？

用户问题：{user_input}

参考资料：
{context_text}

生成的回答：
{response}

请以JSON格式输出：
{{"passed": true, "score": 0.9, "reason": "评估理由"}}"""


def quality_check(state: AgentState) -> dict:
    """轻量模型评估回答质量，写入 quality / retry_count 字段。"""
    retry_count = state.get("retry_count", 0)

    # 重试上限防御：第 3 次无论质量如何都放行，终止循环
    if retry_count >= settings.MAX_RETRY:
        return {
            "quality": QualityResult(passed=True, score=0.5, reason="已达最大重试次数，强制放行"),
            "retry_count": retry_count,
        }

    context_text = "\n\n---\n\n".join(
        chunk["text"] for chunk in state.get("retrieved_contexts", [])
    ) or "（无检索结果）"

    prompt = QUALITY_PROMPT.format(
        user_input=state["user_input"],
        context_text=context_text,
        response=state["generated_response"],
    )

    try:
        result = chat_qwen_json_parsed(prompt)
        quality = QualityResult(
            passed=bool(result["passed"]),
            score=float(result["score"]),
            reason=str(result["reason"]),
        )
    except JSON_TASK_ERRORS as parse_error:
        # 评估自身失败时放行：不能因质检组件故障阻塞回答
        # （含降级链耗尽：两端点同时故障时也必须放行，而非抛异常中断整轮）
        print(f"[quality_check] 质量评估失败，放行: {parse_error}")
        quality = QualityResult(passed=True, score=0.5, reason=f"评估异常放行: {parse_error}")

    if not quality.passed:
        print(f"[quality_check] 评估不通过（第 {retry_count + 1} 次）: {quality.reason}")

    return {"quality": quality, "retry_count": retry_count + 1}


def route_after_quality(state: AgentState) -> str:
    """质量评估后的条件路由：合格走关怀后缀收尾，不合格回退检索重试。"""
    quality = state.get("quality")
    if quality is not None and quality.passed:
        return "care_suffix"
    return "retrieve"
