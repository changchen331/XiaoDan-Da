"""Langfuse 全链路追踪（开关式集成）。

启用条件：``.env`` 中同时配置 ``LANGFUSE_PUBLIC_KEY`` 与 ``LANGFUSE_SECRET_KEY``。
未配置时本模块所有函数均为空操作 / 原样透传，系统行为不受任何影响。

两层追踪覆盖：
1. LLM 调用层：``observe_llm`` 装饰器（Langfuse @observe）注入
   ``agent/llm_clients.py`` 的各 chat 函数，记录调用参数、耗时、返回值
2. 图结构层：``get_langgraph_callbacks`` 返回 CallbackHandler，
   在 graph.invoke 时注入，记录 Agent 各节点的执行轨迹与状态流转

追踪数据用于：Bad Case 回溯（还原某次糟糕回答的完整决策链）、
性能分析（定位延迟瓶颈节点）、每周采样审计。
"""
from config.settings import settings


def get_langgraph_callbacks() -> list:
    """返回 LangGraph 调用所需的 callback handlers。

    :return: Langfuse CallbackHandler 列表；未启用或依赖缺失时返回空列表
    """
    if not settings.langfuse_enabled:
        return []
    try:
        # 延迟导入：未安装 langfuse 时主流程完全不受影响
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except ImportError as import_error:
        print(f"[tracing] langfuse 依赖不可用，跳过图结构追踪: {import_error}")
        return []


def observe_llm(func):
    """LLM 调用追踪装饰器：Langfuse 启用时注入 @observe，否则原样返回函数。

    用法（agent/llm_clients.py）：
        @observe_llm
        def chat_deepseek(...): ...

    装饰后每次调用自动生成一个 Langfuse span（函数名即 span 名），
    记录入参、返回值、耗时与异常。降级路径（DeepSeek → 轻量端点）
    的异常也会被捕获，便于观测降级链的实际触发频率。
    """
    if not settings.langfuse_enabled:
        return func
    try:
        from langfuse import observe
        return observe()(func)
    except ImportError:
        # 依赖缺失时静默透传：追踪是可观测性增强，绝不能阻塞业务调用
        return func
