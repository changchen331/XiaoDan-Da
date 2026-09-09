"""LLM 客户端统一封装与双向降级链。

模型分工（高性价比架构）：
- DeepSeek-V3（云端 API）：回答生成、关怀回复等核心质量环节
- Qwen2.5-14B（本地 vLLM 或任意 OpenAI 兼容端点）：意图路由、质量评估、
  Query 改写、摘要等轻量任务

降级链（双向互备，任一端点故障不影响服务可用）：
- DeepSeek 调用失败 → 自动切换到 Qwen 端点
- Qwen 端点不可用 → 自动切换到 DeepSeek
- 端点不支持 JSON 模式时自动回退为普通调用（Prompt 本身已要求 JSON 输出）

Langfuse 集成：三个 chat 入口函数均挂载 ``@observe_llm`` 装饰器，
启用追踪时每次调用的参数 / 返回值 / 耗时 / 降级事件自动上报，
业务代码零侵入；未启用时装饰器原样透传。
"""
import json

from openai import OpenAI

from config.settings import settings
from observability.tracing import observe_llm


def _build_messages(prompt: str, system: str | None) -> list[dict]:
    """构造 OpenAI 协议的消息列表。

    :param prompt: 用户消息正文
    :param system: 系统提示词（角色设定 / 输出约束），可为空
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_llm(client: OpenAI, model: str, messages: list[dict],
              temperature: float, json_mode: bool = False) -> str:
    """底层单次调用（不降级）：失败直接抛异常，由上层封装决定降级方向。

    :param json_mode: 是否强制 JSON 输出（response_format），用于结构化任务
    """
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def _get_deepseek_client() -> OpenAI:
    """DeepSeek 客户端（惰性创建，进程内复用连接池）。"""
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.DEEPSEEK_TIMEOUT,
    )


def _get_qwen_client() -> OpenAI:
    """轻量任务模型客户端：vLLM / 云端兼容端点均通过该地址接入。"""
    return OpenAI(
        api_key=settings.LOCAL_LLM_API_KEY,
        base_url=settings.LOCAL_LLM_BASE_URL,
        timeout=settings.LOCAL_LLM_TIMEOUT,
    )


@observe_llm
def chat_deepseek(prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    """DeepSeek 调用，失败时自动降级到 Qwen 端点。

    :param prompt: Prompt
    :param system: System Prompt
    :param temperature: 问答场景用 0.3（准确性优先）；关怀回复可用 0.7（表达更自然）
    """
    messages = _build_messages(prompt, system)
    try:
        return _call_llm(_get_deepseek_client(), settings.DEEPSEEK_MODEL,
                         messages, temperature)
    except Exception as deepseek_error:
        # 降级链第一环：核心模型不可用时由轻量端点承接，保证服务不中断
        print(f"[llm_clients] DeepSeek 调用失败，降级到轻量端点: {deepseek_error}")
        return _call_llm(_get_qwen_client(), settings.LOCAL_LLM_MODEL,
                         messages, temperature)


def chat_qwen(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    """轻量任务模型调用，失败时自动降级到 DeepSeek。

    :param prompt: Prompt
    :param system: System Prompt
    :param temperature: 分类 / 评估类任务统一用低温度（0.1）保证确定性
    """
    messages = _build_messages(prompt, system)
    try:
        return _call_llm(_get_qwen_client(), settings.LOCAL_LLM_MODEL,
                         messages, temperature)
    except Exception as qwen_error:
        # 降级链第二环：轻量端点不可用时由 DeepSeek 承接轻量任务
        print(f"[llm_clients] 轻量端点调用失败，降级到 DeepSeek: {qwen_error}")
        return _call_llm(_get_deepseek_client(), settings.DEEPSEEK_MODEL,
                         messages, temperature)


@observe_llm
def chat_qwen_json(prompt: str, system: str | None = None) -> str:
    """轻量任务模型的 JSON 模式调用（意图路由 / 质量评估等结构化场景）。

    三级容错：
    1. 优先使用原生 response_format=json_object
    2. 端点不支持该参数时回退普通调用（Prompt 已要求 JSON 输出）
    3. 轻量端点整体不可用时降级 DeepSeek 的 JSON 模式
    """
    messages = _build_messages(prompt, system)

    # 第一级：原生 JSON 模式
    try:
        return _call_llm(_get_qwen_client(), settings.LOCAL_LLM_MODEL,
                         messages, temperature=0.1, json_mode=True)
    except Exception:
        pass  # 端点可能不支持 response_format，继续尝试回退方案

    # 第二级：普通调用（依靠 Prompt 中的 JSON 输出要求）
    try:
        return _call_llm(_get_qwen_client(), settings.LOCAL_LLM_MODEL,
                         messages, temperature=0.1)
    except Exception as qwen_error:
        # 第三级：轻量端点整体不可用，DeepSeek JSON 模式兜底
        print(f"[llm_clients] 轻量端点不可用，JSON 任务降级到 DeepSeek: {qwen_error}")
        return _call_llm(_get_deepseek_client(), settings.DEEPSEEK_MODEL,
                         messages, temperature=0.1, json_mode=True)


def parse_json_response(raw: str) -> dict:
    """解析 LLM 返回的 JSON 文本，容忍 Markdown 代码块包裹等常见噪声。

    解析策略：直接 json.loads 失败时，截取首个 "{"
    到最后一个 "}" 之间的片段重试（模型偶尔在 JSON 前后附加说明文字）。

    :raises ValueError: 内容中确实不含合法 JSON 对象时抛出，由调用方兜底
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"响应中未找到合法 JSON: {raw[:100]}")
