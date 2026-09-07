"""节点：检索——简单问答分支，对改写后的 query 执行混合检索 + 重排。

本节点是模块一检索服务在 Agent 图中的接入点：
改写 query → BGE-M3 双路向量化 → 元数据过滤（身份 + 学期时效）
→ Milvus 混合检索（dense/sparse 各 top20，RRF 融合）→ Reranker 精排取 top5。
"""
from agent.state import AgentState
from config.settings import settings, get_current_semester


def build_filter_expr(user_profile: dict) -> str:
    """构造元数据过滤表达式：按用户身份与当前学期过滤知识库。

    身份过滤：文档 target_audience 为"全部"或与用户角色一致时可见
    时效过滤：文档 valid_semester 为"长期有效"或等于当前学期时可见
    （例如 2024 年的选课通知不应出现在 2026 年的检索结果中）

    :param user_profile: 用户画像，role 字段取值
        本科生 / 研究生 / 留学生 / 教职工；未知角色不做身份过滤
    """
    role = user_profile.get("role", "全部")
    semester = get_current_semester()

    # 未知角色（如教职工）放宽为全员可见，避免过滤条件过严导致空结果
    audiences = [role, "全部"] if role in ("本科生", "研究生", "留学生") else ["全部"]
    audience_list = ", ".join(f'"{audience}"' for audience in audiences)

    return (
        f'metadata["target_audience"] in [{audience_list}] and '
        f'metadata["valid_semester"] in ["{semester}", "长期有效"]'
    )


def retrieve(state: AgentState) -> dict:
    """执行混合检索 + 重排，结果写入 retrieved_contexts。

    检索失败时返回空结果集：下游 generate 节点对空上下文有明确的
    "无法确定"应答策略，保证服务不因中间件故障而中断。
    """
    from knowledge_base.retrieval import get_retrieval_service

    query = state["intent"].rewritten_query
    filter_expr = build_filter_expr(state.get("user_profile", {}))

    try:
        service = get_retrieval_service()
        candidates = service.hybrid_search(query, filter_expr=filter_expr)
        reranked = service.rerank(query, candidates, top_k=settings.RERANK_TOP_K)
    except Exception as retrieval_error:
        print(f"[retrieve] 检索服务异常，返回空结果集: {retrieval_error}")
        reranked = []

    return {"retrieved_contexts": reranked, "faq_hit": False}
