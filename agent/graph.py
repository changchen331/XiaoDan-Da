"""LangGraph 图编排：声明式定义 11 个节点与全部边，流程完全可控、可解释。

执行路径：
    START → emotion_detect（每轮必经）
              ├─ 高危   → report → care_response → END
              └─ 正常   → intent_route
                            ├─ 简单问答 → retrieve ─────────────┐
                            ├─ 复杂查询 → plan_and_retrieve ────┤→ generate → quality_check
                            ├─ FAQ     → faq_match ─┬─ 命中 ──→ memory_write → END
                            │                      └─ 未命中 ─→ generate（走上方路径）
                            └─ 闲聊越界 → chitchat_response → END
    quality_check ─ 合格 → memory_write → END
                 └─ 不合格 → retrieve（重试，最多 2 次后强制放行）

短期记忆：PostgresSaver 作为 Checkpointer（thread_id 即 session_id），
同一会话的多轮输入自动携带历史状态；数据库不可用时自动降级为
无记忆模式（由调用方显式传入 conversation_history 补偿）。
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    care_response,
    chitchat_response,
    emotion_detect,
    faq_match,
    generate,
    intent_route,
    memory_write,
    plan_and_retrieve,
    quality_check,
    report,
    retrieve,
)
from agent.nodes.emotion_detect import route_after_emotion
from agent.nodes.faq_match import route_after_faq
from agent.nodes.intent_route import route_after_intent
from agent.nodes.quality_check import route_after_quality
from agent.state import AgentState
from config.settings import settings
from observability.tracing import get_langgraph_callbacks

# ===== 进程级单例 =====
# Checkpointer 连接（PostgreSQL 长连接，进程生命周期内复用）
_checkpointer = None
_checkpointer_initialized = False
# 编译后的图应用（含节点与边的一次性构建产物，线程安全可复用）
_compiled_app = None


def _get_checkpointer():
    """获取 PostgresSaver 短期记忆组件（惰性初始化，全局唯一）。

    setup() 幂等建表（langgraph 内部的 checkpoint 表族）。
    PostgreSQL 不可达时返回 None，图以无记忆模式编译，
    由 invoke 调用方传入 conversation_history 补偿多轮上下文。
    """
    global _checkpointer, _checkpointer_initialized
    if not _checkpointer_initialized:
        _checkpointer_initialized = True
        try:
            import psycopg2
            from langgraph.checkpoint.postgres import PostgresSaver

            connection = psycopg2.connect(settings.postgres_dsn)
            checkpointer = PostgresSaver(connection)
            checkpointer.setup()
            _checkpointer = checkpointer
        except Exception as db_error:
            print(f"[graph] Checkpointer 初始化失败，本轮进程降级为无短期记忆: {db_error}")
    return _checkpointer


def build_graph():
    """构建并编译小旦答 Agent 状态图。

    :return: 编译后的 LangGraph 应用；配置了 Checkpointer 时
        同一 thread_id 的多次 invoke 自动共享会话状态
    """
    graph = StateGraph(AgentState)

    # ===== 添加全部 11 个节点 =====
    graph.add_node("emotion_detect", emotion_detect)     # 情绪检测（每轮必经）
    graph.add_node("report", report)                     # 高危上报（写库 + 邮件）
    graph.add_node("care_response", care_response)       # 高危关怀回复
    graph.add_node("intent_route", intent_route)         # 意图分类 + Query 改写
    graph.add_node("retrieve", retrieve)                 # 简单问答：单次混合检索
    graph.add_node("plan_and_retrieve", plan_and_retrieve)  # 复杂查询：拆解检索
    graph.add_node("faq_match", faq_match)               # FAQ 高置信度匹配
    graph.add_node("chitchat_response", chitchat_response)  # 闲聊越界兜底
    graph.add_node("generate", generate)                 # 基于上下文生成回答
    graph.add_node("quality_check", quality_check)      # 在线质量评估
    graph.add_node("memory_write", memory_write)         # 长期记忆写入

    # ===== 入口：先做情绪检测（安全优先于一切问答逻辑）=====
    graph.add_edge(START, "emotion_detect")

    # 情绪分流：高危 → 上报分支；其余 → 正常问答
    graph.add_conditional_edges(
        "emotion_detect", route_after_emotion,
        {"report": "report", "intent_route": "intent_route"},
    )

    # 高危分支：上报 → 关怀回复 → 结束（阻断正常问答，暂缓回答原始问题）
    graph.add_edge("report", "care_response")
    graph.add_edge("care_response", END)

    # 意图分流：四条处理分支
    graph.add_conditional_edges(
        "intent_route", route_after_intent,
        {
            "retrieve": "retrieve",
            "plan_and_retrieve": "plan_and_retrieve",
            "faq_match": "faq_match",
            "chitchat_response": "chitchat_response",
        },
    )

    # 三条检索分支汇入生成节点
    graph.add_edge("retrieve", "generate")
    graph.add_edge("plan_and_retrieve", "generate")

    # FAQ 分支：命中标准答案直接输出（跳过生成与质检），未命中走生成
    graph.add_conditional_edges(
        "faq_match", route_after_faq,
        {"memory_write": "memory_write", "generate": "generate"},
    )

    # 闲聊越界：直接兜底回复结束（不检索不生成）
    graph.add_edge("chitchat_response", END)

    # 生成 → 在线质量评估
    graph.add_edge("generate", "quality_check")

    # 质量分流：合格写记忆收尾；不合格回退检索重试（受 retry_count 上限保护）
    graph.add_conditional_edges(
        "quality_check", route_after_quality,
        {"memory_write": "memory_write", "retrieve": "retrieve"},
    )

    # 记忆写入 → 结束
    graph.add_edge("memory_write", END)

    # 编译：PostgreSQL 可用时挂载 Checkpointer（短期记忆）
    checkpointer = _get_checkpointer()
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def get_app():
    """获取编译后的图应用（进程级单例，避免重复构建）。"""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_graph()
    return _compiled_app


def invoke(user_input: str, user_id: str = "anonymous",
           session_id: str = "default", conversation_history: list | None = None,
           user_profile: dict | None = None) -> dict:
    """单次调用 Agent 的完整流水线。

    :param user_input: 用户原始输入
    :param user_id: 用户内部 ID（脱敏标识）
    :param session_id: 会话 ID，同时作为 Checkpointer 的 thread_id，
        同一 session_id 的多次调用自动共享短期记忆
    :param conversation_history: 显式传入的对话历史（Checkpointer 降级时的补偿通道）
    :param user_profile: 用户画像 {"role": 本科生/研究生/留学生/教职工}
    :return: 最终 State，关键字段为 final_response（输出回答）与 emotion（情绪检测）
    """
    initial_state: AgentState = {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": conversation_history or [],
        "user_profile": user_profile or {"role": "本科生"},
        "retry_count": 0,
    }

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.RECURSION_LIMIT,   # 递归上限，兜底防死循环
        "callbacks": get_langgraph_callbacks(),        # Langfuse 启用时注入全链路追踪
    }

    return get_app().invoke(initial_state, config)
