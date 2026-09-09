"""节点：检索——统一入口：FAQ 预查 + 混合检索 + 重排。

两条路径合流于此（简单问答与 FAQ 意图均路由到本节点）：
快路径：FAQ 专用索引向量匹配（约 10ms）→ 相似度 > 0.85 →
        Qwen 轻量校验（答案适配 + 语言匹配）→ 通过则标准答案直返，
        跳过生成与质检（响应延迟秒级 → 百毫秒级）。
慢路径：BGE-M3 双路向量化 → 元数据过滤（身份 + 学期时效）→
        Milvus 混合检索（dense/sparse 各 top20，RRF 融合）→
        Reranker 精排取 top5，交由 generate 生成。

快路径不依赖意图分类的准确性：即使高频问题被误分类为"简单问答"，
预查依然先于混合检索执行，秒回能力不受分类器影响；
质量评估不合格回退到本节点重试时同样先过 FAQ 预查。
"""
from agent.faq_verify import verify_faq_match
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


def _faq_precheck(query: str, response_language: str) -> dict | None:
    """FAQ 快路径预查：索引匹配 + 轻量校验。

    :param query: 改写后的检索 query
    :param response_language: 用户期望的回复语言
    :return: 可直返的命中结果 {"question", "answer", "score"}；
        未命中 / 阈值不足 / 校验拒绝 / 索引异常均返回 None（走慢路径）
    """
    from knowledge_base.faq import get_faq_index

    try:
        matched = get_faq_index().match(query)
    except Exception as faq_error:
        # FAQ 索引异常（Milvus 不可达等）：静默降级到通用检索，流程不中断
        print(f"[retrieve] FAQ 预查异常，降级通用检索: {faq_error}")
        return None

    if matched is None:
        return None

    # 高阈值命中后做轻量校验：防"问 A 答 B"与语言不匹配的直返
    if not verify_faq_match(query, matched["question"], matched["answer"],
                            response_language):
        print(f"[retrieve] FAQ 命中被校验拒绝（相似度 {matched['score']:.3f}）: "
              f"{matched['question']}")
        return None

    print(f"[retrieve] FAQ 命中（相似度 {matched['score']:.3f}）: {matched['question']}")
    return matched


def retrieve(state: AgentState) -> dict:
    """检索入口节点：先 FAQ 预查，未命中走混合检索 + 重排。

    检索失败时返回空结果集：下游 generate 节点对空上下文有明确的
    "无法确定"应答策略，保证服务不因中间件故障而中断。
    """
    from knowledge_base.retrieval import get_retrieval_service

    query = state["intent"].rewritten_query
    response_language = state.get("response_language", "中文")

    # ===== 快路径：FAQ 专用索引预查 =====
    matched = _faq_precheck(query, response_language)
    if matched is not None:
        return {
            "faq_hit": True,
            "generated_response": matched["answer"],  # 标准答案直接作为生成结果
            "retrieved_contexts": [],                 # FAQ 命中无检索上下文
        }

    # ===== 慢路径：通用混合检索 =====
    filter_expr = build_filter_expr(state.get("user_profile", {}))

    try:
        service = get_retrieval_service()
        candidates = service.hybrid_search(query, filter_expr=filter_expr)
        reranked = service.rerank(query, candidates, top_k=settings.RERANK_TOP_K)
    except Exception as retrieval_error:
        print(f"[retrieve] 检索服务异常，返回空结果集: {retrieval_error}")
        reranked = []

    return {"retrieved_contexts": reranked, "faq_hit": False}


def route_after_retrieve(state: AgentState) -> str:
    """检索后的条件路由：FAQ 命中走关怀后缀直返，未命中进生成流程。"""
    if state.get("faq_hit"):
        return "care_suffix"
    return "generate"
