"""节点：FAQ 匹配——优先从 FAQ 专用索引精确命中标准答案。

高置信度命中的价值：答案由管理员审定，直接返回即保证权威性，
同时跳过 LLM 生成与质量评估环节，响应延迟从秒级降到百毫秒级。

命中判定：问题向量最高相似度 > 0.85（阈值可配）。
未命中：无缝降级到通用 RAG 检索流程（retrieve 节点），
对用户而言两条路径的体验一致。
"""
from agent.nodes.retrieve import build_filter_expr, retrieve
from agent.state import AgentState


def faq_match(state: AgentState) -> dict:
    """在 FAQ 专用 Collection 中做向量匹配。

    :return: 命中时写入 faq_hit=True 与标准答案（generated_response），
        由图的条件边路由到 memory_write 直接输出；
        未命中时执行通用检索并返回其结果。
    """
    from knowledge_base.faq import get_faq_index

    query = state["intent"].rewritten_query

    try:
        matched = get_faq_index().match(query)
    except Exception as faq_error:
        # FAQ 索引异常（Milvus 不可达等）：直接走通用检索，流程不中断
        print(f"[faq_match] FAQ 索引异常，降级通用检索: {faq_error}")
        matched = None

    if matched is not None:
        print(f"[faq_match] FAQ 命中（相似度 {matched['score']:.3f}）: {matched['question']}")
        return {
            "faq_hit": True,
            "generated_response": matched["answer"],   # 标准答案直接作为生成结果
            "retrieved_contexts": [],                    # FAQ 命中无检索上下文
        }

    # 未命中：降级到通用混合检索（保持与简单问答一致的处理）
    return retrieve(state)


def route_after_faq(state: AgentState) -> str:
    """FAQ 匹配后的条件路由：命中直接写记忆结束，未命中走生成流程。"""
    if state.get("faq_hit"):
        return "memory_write"
    return "generate"
