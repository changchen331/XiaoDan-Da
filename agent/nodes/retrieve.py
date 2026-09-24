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

质检重试（retry_count > 0）会**改变检索输入**：候选池随重试次数翻倍、
重排片段逐次 +2、并去掉人群过滤（保留学期时效过滤）——否则重试拿到
与首次完全相同的确定性结果，生成不可能有信息增益。
"""
from agent.faq_verify import verify_faq_match
from agent.state import AgentState
from config.settings import get_current_semester, settings


def build_filter_expr(user_profile: dict, include_audience: bool = True) -> str:
    """构造元数据过滤表达式：按用户身份与当前学期过滤知识库。

    身份过滤：文档 target_audience 为"全部"或与用户角色一致时可见
    时效过滤：文档 valid_semester 为"长期有效"或等于当前学期时可见
    （例如 2024 年的选课通知不应出现在 2026 年的检索结果中）

    :param user_profile: 用户画像，role 字段取值
        本科生 / 研究生 / 留学生 / 教职工；未知角色不做身份过滤
    :param include_audience: 是否保留身份过滤。质检重试时置 False（放宽召回）：
        答案可能恰好在受众不匹配的文档里；而时效过滤必须保留——
        把过期的选课通知交给模型，误导学生的代价高于答不上来
    """
    semester = get_current_semester()
    semester_condition = f'metadata["valid_semester"] in ["{semester}", "长期有效"]'
    if not include_audience:
        return semester_condition

    role = user_profile.get("role", "全部")
    # 未知角色（如教职工）放宽为全员可见，避免过滤条件过严导致空结果
    audiences = [role, "全部"] if role in ("本科生", "研究生", "留学生") else ["全部"]
    audience_list = ", ".join(f'"{audience}"' for audience in audiences)

    return f'metadata["target_audience"] in [{audience_list}] and {semester_condition}'


def _faq_precheck(
    query: str, response_language: str, user_input: str | None = None
) -> dict | None:
    """FAQ 快路径预查：索引匹配 + 轻量校验。

    :param query: 改写后的检索 query（规范化后的问法，利于匹配规范表述）
    :param response_language: 用户期望的回复语言
    :param user_input: 用户**原话**。它与改写 query 各有用处，故**两条都参与匹配、
        取相似度更高者**：实测"论文到底要交到哪儿去？"原话命中 0.953、
        改写后只有 0.855（贴着 0.85 阈值，命中与否随机）；
        而"云邮箱能发多大的附件？"两条分别 0.966 / 0.863。
        只用一个会让快路径在阈值边缘抖动，故两个都试（多一次向量检索，成本极低）。
    :return: 可直返的命中结果 {"question", "answer", "score"}；
        未命中 / 阈值不足 / 校验拒绝 / 索引异常均返回 None（走慢路径）
    """
    from knowledge_base.faq import get_faq_index

    matched: dict | None = None
    for candidate in {user_input, query} - {None, ""}:
        try:
            hit = get_faq_index().match(candidate)
        except Exception as faq_error:
            # FAQ 索引异常（Milvus 不可达等）：静默降级到通用检索，流程不中断
            print(f"[retrieve] FAQ 预查异常，降级通用检索: {faq_error}")
            return None
        if hit is not None and (matched is None or hit["score"] > matched["score"]):
            matched = hit

    if matched is None:
        return None

    # 高阈值命中后做轻量校验：防"问 A 答 B"与语言不匹配的直返。
    # 校验用**用户原话**——改写会把问法放宽（实测"…是什么？"被改写成
    # "…介绍及使用指南"），拿它校验会把简短的标准答案判成"答不全"而误拒
    if not verify_faq_match(
        user_input or query, matched["question"], matched["answer"], response_language
    ):
        print(f"[retrieve] FAQ 命中被校验拒绝（相似度 {matched['score']:.3f}）: "
              f"{matched['question']}")
        return None

    print(f"[retrieve] FAQ 命中（相似度 {matched['score']:.3f}）: {matched['question']}")
    return matched


def retrieve(state: AgentState) -> dict:
    """检索入口节点：先 FAQ 预查，未命中走混合检索 + 重排。

    质检重试（retry_count > 0）时放宽检索：候选池随重试次数翻倍、
    重排片段逐次 +2、去掉人群过滤——保证重试的输入与首次不同，
    否则同一份确定性结果会让重试永远拿不到新信息。

    检索失败时返回空结果集：下游 generate 节点对空上下文有明确的
    "无法确定"应答策略，保证服务不因中间件故障而中断。
    """
    from knowledge_base.retrieval import get_retrieval_service

    query = state["intent"].rewritten_query
    response_language = state.get("response_language", "中文")
    retry_count = state.get("retry_count", 0)

    # ===== 快路径：FAQ 专用索引预查 =====
    # 匹配用改写 query（召回），校验用用户原话（判定是否答得其所问），见 _faq_precheck
    matched = _faq_precheck(query, response_language, state.get("user_input"))
    if matched is not None:
        return {
            "faq_hit": True,
            "generated_response": matched["answer"],  # 标准答案直接作为生成结果
            "retrieved_contexts": [],  # FAQ 命中无检索上下文
        }

    # ===== 慢路径：通用混合检索（重试时逐级放宽）=====
    filter_expr = build_filter_expr(
        state.get("user_profile", {}), include_audience=retry_count == 0
    )
    hybrid_top_k = settings.HYBRID_TOP_K * (retry_count + 1)
    rerank_top_k = settings.RERANK_TOP_K + 2 * retry_count
    if retry_count > 0:
        print(
            f"[retrieve] 质检重试第 {retry_count} 次：放宽检索"
            f"（去人群过滤、候选 {hybrid_top_k}、重排取 {rerank_top_k}）"
        )

    try:
        service = get_retrieval_service()
        candidates = service.hybrid_search(
            query, filter_expr=filter_expr, top_k=hybrid_top_k
        )
        reranked = service.rerank(query, candidates, top_k=rerank_top_k)
    except Exception as retrieval_error:
        print(f"[retrieve] 检索服务异常，返回空结果集: {retrieval_error}")
        reranked = []

    return {"retrieved_contexts": reranked, "faq_hit": False}


def route_after_retrieve(state: AgentState) -> str:
    """检索后的条件路由：FAQ 命中走关怀后缀直返，未命中进生成流程。"""
    if state.get("faq_hit"):
        return "care_suffix"
    return "generate"
