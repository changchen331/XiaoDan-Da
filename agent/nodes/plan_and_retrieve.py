"""节点：规划检索——复杂查询分支，拆解子查询分别检索后合并。

复杂查询的特征：单一检索无法覆盖全部信息需求。
例如"留学生选课和本科生有什么区别"需要分别检索
"留学生选课政策"与"本科生选课政策"两个独立信息源。

执行流程：
1. 轻量模型将复杂问题拆解为 2-4 个独立的子查询
2. 每个子查询独立走混合检索（top10）
3. 按 chunk 文本哈希合并去重（同一文档可能被多个子查询命中）
4. Reranker 以原始问题为锚做全局重排，取 top8
   （比简单问答的 top5 略宽：复杂问题需要更多素材做综合）
"""
from agent.query_filter import build_filter_expr
from agent.state import AgentState
from config.settings import settings
from infra.llm_clients import JSON_TASK_ERRORS, chat_qwen_json_parsed

# 复杂查询的拆解指令：子查询必须各自独立可检索
DECOMPOSE_PROMPT = """请将以下复杂问题拆解为2-4个独立的子查询。
每个子查询应该是一个可以单独检索的简单问题，子查询合起来能覆盖原始问题的全部信息需求。

原始问题：{query}

请以JSON格式输出：
{{"sub_queries": ["子查询1", "子查询2"]}}"""


def plan_and_retrieve(state: AgentState) -> dict:
    """复杂查询：拆解 → 逐个检索 → 合并去重 → 全局重排取 top8。"""
    from knowledge_base.retrieval import get_retrieval_service

    query = state["intent"].rewritten_query
    filter_expr = build_filter_expr(state.get("user_profile", {}))

    # 第一步：拆解子查询（拆解失败则退化为单查询处理，流程不中断）
    sub_queries = _decompose(query)
    print(f"[plan_and_retrieve] 拆解出 {len(sub_queries)} 个子查询: {sub_queries}")

    try:
        service = get_retrieval_service()

        # 第二步：每个子查询独立召回
        all_candidates: list = []
        for sub_query in sub_queries:
            all_candidates.extend(
                service.hybrid_search(sub_query, filter_expr=filter_expr, top_k=10)
            )

        # 第三步：按 chunk 文本去重（多子查询命中同一文档时只保留一份）
        unique_candidates: list = []
        seen_texts: set = set()
        for candidate in all_candidates:
            text_key = hash(candidate["text"])
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_candidates.append(candidate)

        # 第四步：以原始问题为锚做全局重排（子查询各自的分数不可比，必须重排）
        reranked = service.rerank(query, unique_candidates, top_k=settings.RERANK_TOP_K + 3)
    except Exception as retrieval_error:
        print(f"[plan_and_retrieve] 检索服务异常，返回空结果集: {retrieval_error}")
        reranked = []

    return {"retrieved_contexts": reranked, "faq_hit": False}


def _decompose(query: str) -> list:
    """调用轻量模型拆解子查询。

    :return: 子查询列表；拆解失败时回退为 [原始query]（退化为简单检索）
    """
    try:
        sub_queries = chat_qwen_json_parsed(DECOMPOSE_PROMPT.format(query=query))[
            "sub_queries"
        ]
        if isinstance(sub_queries, list) and sub_queries:
            return [str(sq) for sq in sub_queries]
    except JSON_TASK_ERRORS as parse_error:
        print(f"[plan_and_retrieve] 子查询拆解失败，按单查询处理: {parse_error}")
    return [query]
