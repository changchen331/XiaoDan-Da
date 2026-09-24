"""LLM 客户端统一封装与双向降级链。

模型分工（高性价比架构）：
- DeepSeek（云端 API，``DEEPSEEK_*``）：回答生成、关怀回复等核心质量环节
- 轻量任务模型（``LIGHT_LLM_*``，主用）：意图路由、质量评估、Query 改写、
  摘要、FAQ 校验。端点可指向云端兼容服务，也可指向本地推理服务
- 本地兜底模型（``FALLBACK_LLM_*``，可选）：云端端点**全部不可用**时
  接管轻量任务，恢复真正的离线可用性；默认关闭，选定模型后开启

降级链（逐级下探，写出「降级也用不了怎么办」的答案）：
- 轻量任务：轻量端点 → DeepSeek → 本地兜底 → ``LLMUnavailableError``
- 回答生成：DeepSeek → 轻量端点 → ``LLMUnavailableError``
  （生成与高危关怀**刻意不接本地小模型**：这两项对生成质量要求最高，
  宁可由调用方给保守兜底文案，也不要用小模型编出不可靠的制度性内容）
- 端点不支持 JSON 模式时自动回退为普通调用（Prompt 本身已要求 JSON 输出）
- 思考模式按角色开关（``DEEPSEEK_THINKING`` / ``LIGHT_LLM_THINKING``，**默认关闭**）：
  现行型号默认开启思考，代价是 temperature 被厂商改写（千问：小于 0.6 的值自动调整
  为 0.6）且思考 token 按输出价计费，而本项目要的是低温度下的确定性输出
- 端点不可达 / 鉴权失败时**跳过一次注定失败的重试**，直接进入下一跳

Langfuse 集成：三个 chat 入口函数均挂载 ``@observe_llm`` 装饰器，
启用追踪时每次调用的参数 / 返回值 / 耗时 / 降级事件自动上报，
业务代码零侵入；未启用时装饰器原样透传。
"""

import json

from openai import APIConnectionError, AuthenticationError, OpenAI

from config.settings import settings
from observability.tracing import observe_llm


class LLMUnavailableError(RuntimeError):
    """降级链已耗尽：所有候选端点均调用失败。

    存在的意义是让「所有端点都挂了」成为一个**可判定的失败信号**。
    此前降级链的最后一跳没有兜底，openai 的原生异常会直接穿透节点、
    把整轮请求打挂（HTTP 500）——而各节点其实都写了「失败时降级」的意图，
    只是捕获的异常类型是按预期写的，抓不到实际抛出的 `OpenAIError`。

    调用方约定：捕获本异常并降级到**确定性兜底逻辑**
    （如意图路由退回「简单问答」、质检退回放行）。
    """


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


def _deepseek_extra_body() -> dict:
    """DeepSeek 侧的思考模式开关请求片段（默认关闭）。

    DeepSeek 用 ``thinking: {"type": "enabled" | "disabled"}`` 控制，
    ``disabled`` 时模型直接作答、不再产出思考内容。
    """
    thinking_type = "enabled" if settings.DEEPSEEK_THINKING else "disabled"
    return {"thinking": {"type": thinking_type}}


def _light_extra_body() -> dict:
    """轻量端点（千问 OpenAI 兼容模式）侧的思考模式开关请求片段（默认关闭）。

    千问兼容模式用扁平的 ``enable_thinking`` 布尔值控制，与 DeepSeek 的嵌套写法不同，
    故各自封装、避免调用处写错参数名。**本组默认关闭是硬要求**：
    思考模式下厂商会把 temperature 强制改写（文档：传入更小值自动调整为 0.6），
    而本组任务的设计前提是 0.1 的确定性输出。
    """
    return {"enable_thinking": settings.LIGHT_LLM_THINKING}


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    json_mode: bool = False,
    extra_body: dict | None = None,
) -> str:
    """底层单次调用（不降级）：失败直接抛异常，由上层封装决定降级方向。

    :param json_mode: 是否强制 JSON 输出（response_format），用于结构化任务
    :param extra_body: 厂商扩展参数（如思考模式开关），按端点所属模型族由调用方给出
    """
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


# 客户端缓存：OpenAI 对象内部持有 httpx 连接池，必须在进程内复用它才有意义。
# 此前每次调用都新建，等于每次请求都重建连接池、丢掉 keep-alive
# （函数名与注释都宣称了复用语义，实现没有兑现）。
_deepseek_client: OpenAI | None = None
_light_llm_client: OpenAI | None = None
_fallback_llm_client: OpenAI | None = None


def _get_deepseek_client() -> OpenAI:
    """DeepSeek 客户端（惰性创建，进程内复用连接池）。"""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=settings.DEEPSEEK_TIMEOUT,
        )
    return _deepseek_client


def _get_light_llm_client() -> OpenAI:
    """轻量任务模型客户端（主用）：任意 OpenAI 兼容端点，云或本地均可。"""
    global _light_llm_client
    if _light_llm_client is None:
        _light_llm_client = OpenAI(
            api_key=settings.LIGHT_LLM_API_KEY,
            base_url=settings.LIGHT_LLM_BASE_URL,
            timeout=settings.LIGHT_LLM_TIMEOUT,
        )
    return _light_llm_client


def _get_fallback_llm_client() -> OpenAI:
    """本地兜底模型客户端（仅在云端端点全部失败时使用）。"""
    global _fallback_llm_client
    if _fallback_llm_client is None:
        _fallback_llm_client = OpenAI(
            api_key=settings.FALLBACK_LLM_API_KEY,
            base_url=settings.FALLBACK_LLM_BASE_URL,
            timeout=settings.FALLBACK_LLM_TIMEOUT,
        )
    return _fallback_llm_client


def _try_local_fallback(messages: list[dict], temperature: float) -> str:
    """最后一跳：交给本地兜底模型（未启用或未配置模型时抛 LLMUnavailableError）。

    本地兜底是"云端全挂"时唯一的出路，因此这里**不再向下降级**——
    失败即抛可判定的异常，由调用方走各自的确定性兜底逻辑。

    本地小模型无思考模式（也不支持该参数），故不传 extra_body：
    给不认识的端点硬塞厂商扩展参数只会换来一个 400。
    """
    if not settings.FALLBACK_LLM_ENABLED or not settings.FALLBACK_LLM_MODEL:
        raise LLMUnavailableError(
            "本地兜底模型未启用（FALLBACK_LLM_ENABLED=false 或 FALLBACK_LLM_MODEL 为空）"
        )
    print(f"[llm_clients] 尝试本地兜底模型: {settings.FALLBACK_LLM_MODEL}")
    return _call_llm(
        _get_fallback_llm_client(),
        settings.FALLBACK_LLM_MODEL,
        messages,
        temperature,
    )


@observe_llm
def chat_deepseek(
    prompt: str, system: str | None = None, temperature: float = 0.3
) -> str:
    """DeepSeek 调用，失败时自动降级到轻量端点。

    **本入口不接本地兜底**：它承载回答生成与高危关怀回复（非轻量任务），
    而本地小模型的生成质量不足以承担这两项，宁可由调用方给保守兜底文案。

    :param prompt: Prompt
    :param system: System Prompt
    :param temperature: 问答场景用 0.3（准确性优先）；关怀回复可用 0.7（表达更自然）
    """
    messages = _build_messages(prompt, system)
    try:
        return _call_llm(
            _get_deepseek_client(),
            settings.DEEPSEEK_MODEL,
            messages,
            temperature,
            extra_body=_deepseek_extra_body(),
        )
    except Exception as deepseek_error:
        # 降级链第一环：核心模型不可用时由轻量端点承接，保证服务不中断
        print(f"[llm_clients] DeepSeek 调用失败，降级到轻量端点: {deepseek_error}")
        try:
            return _call_llm(
                _get_light_llm_client(),
                settings.LIGHT_LLM_MODEL,
                messages,
                temperature,
                extra_body=_light_extra_body(),
            )
        except Exception as light_error:
            # 降级链终点：两条路都不通，抛出可判定的信号而非原生异常
            raise LLMUnavailableError(
                f"DeepSeek 与轻量端点均调用失败：{deepseek_error}；{light_error}"
            ) from light_error


@observe_llm
def chat_qwen(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    """轻量任务模型调用，失败时依次降级：DeepSeek → 本地兜底模型。

    :param prompt: Prompt
    :param system: System Prompt
    :param temperature: 分类 / 评估类任务统一用低温度（0.1）保证确定性
    """
    messages = _build_messages(prompt, system)
    try:
        return _call_llm(
            _get_light_llm_client(),
            settings.LIGHT_LLM_MODEL,
            messages,
            temperature,
            extra_body=_light_extra_body(),
        )
    except Exception as light_error:
        # 降级链第二环：轻量端点不可用时由 DeepSeek 承接轻量任务
        print(f"[llm_clients] 轻量端点调用失败，降级到 DeepSeek: {light_error}")
        try:
            return _call_llm(
                _get_deepseek_client(),
                settings.DEEPSEEK_MODEL,
                messages,
                temperature,
                extra_body=_deepseek_extra_body(),
            )
        except Exception as deepseek_error:
            # 降级链第三环：云端全挂 → 交给本地兜底模型（离线可用性的底线）
            print(f"[llm_clients] DeepSeek 也不可用，转向本地兜底: {deepseek_error}")
            try:
                return _try_local_fallback(messages, temperature)
            except Exception as fallback_error:
                raise LLMUnavailableError(
                    f"轻量端点、DeepSeek 与本地兜底均失败："
                    f"{light_error}；{deepseek_error}；{fallback_error}"
                ) from fallback_error


@observe_llm
def chat_qwen_json(prompt: str, system: str | None = None) -> str:
    """轻量任务模型的 JSON 模式调用（意图路由 / 质量评估等结构化场景）。

    容错顺序：
    1. 优先使用原生 response_format=json_object
    2. 端点不支持该参数时回退普通调用（Prompt 已要求 JSON 输出）
    3. 轻量端点整体不可用时降级 DeepSeek 的 JSON 模式
    4. 云端全挂时交给本地兜底模型（同样用 JSON 模式）

    :raises LLMUnavailableError: 连本地兜底也失败时（降级链已耗尽）
    """
    return _call_json_chain(_build_messages(prompt, system))[0]


@observe_llm
def chat_qwen_json_with_source(
    prompt: str, system: str | None = None
) -> tuple[str, str]:
    """同 :func:`chat_qwen_json`，但额外回报**实际应答的端点来源**。

    返回 ``(内容, 来源)``，来源取 ``"云端"`` / ``"本地"``。

    **为什么需要它**：情绪语义判别层在"本地兜底模型"应答时必须收紧策略——
    本地小模型的指向抽取准确率实测仅 62%，只允许它升级风险等级、不允许降级
    （否则降级带来的漏报率直接等于它的错判率）。而"是否本地应答"只有降级链
    自己知道，调用方无法从返回内容反推。
    """
    return _call_json_chain(_build_messages(prompt, system))


def _call_json_chain(messages: list[dict]) -> tuple[str, str]:
    """JSON 模式的降级链（**唯一实现**），返回 ``(内容, 来源)``。

    来源：``"云端"`` = 轻量端点或 DeepSeek 应答；``"本地"`` = 本地兜底模型应答。

    把 JSON 入口的降级链收敛到这一处，是因为三个 LLM 入口此前各写一套、行为已经漂移。
    本函数承载全部跳级逻辑：
    端点不可达 / 鉴权失败时**跳过一次注定失败的重试**（否则要白等两个超时周期），
    其余错误才值得换普通调用再试一次。
    """
    light_error: Exception

    # 第一级：原生 JSON 模式
    try:
        return (
            _call_llm(
                _get_light_llm_client(),
                settings.LIGHT_LLM_MODEL,
                messages,
                temperature=0.1,
                json_mode=True,
                extra_body=_light_extra_body(),
            ),
            "云端",
        )
    except (APIConnectionError, AuthenticationError) as reach_error:
        # 端点不可达 / 鉴权失败：**重试同一端点只会再白等一个完整超时周期**，
        # 因此跳过第二级直接降级。此前这里对不可达端点重试一次，
        # 端点完全挂掉时要先等满两个超时（实测 15s × 2）才走降级
        light_error = reach_error
    except Exception:
        # 其余失败（典型是端点不支持 response_format）才值得换普通调用再试一次
        try:
            return (
                _call_llm(
                    _get_light_llm_client(),
                    settings.LIGHT_LLM_MODEL,
                    messages,
                    temperature=0.1,
                    extra_body=_light_extra_body(),
                ),
                "云端",
            )
        except Exception as plain_error:
            light_error = plain_error

    # 第三级：轻量端点整体不可用，DeepSeek JSON 模式兜底
    print(f"[llm_clients] 轻量端点不可用，JSON 任务降级到 DeepSeek: {light_error}")
    try:
        return (
            _call_llm(
                _get_deepseek_client(),
                settings.DEEPSEEK_MODEL,
                messages,
                temperature=0.1,
                json_mode=True,
                extra_body=_deepseek_extra_body(),
            ),
            "云端",
        )
    except Exception as deepseek_error:
        # 第四级：云端全挂 → 本地兜底模型（JSON 模式）
        print(
            f"[llm_clients] DeepSeek 也不可用，JSON 任务转向本地兜底: {deepseek_error}"
        )
        try:
            return _try_local_fallback(messages, 0.1), "本地"
        except Exception as fallback_error:
            # 降级链终点：转成可判定的异常，避免 openai 原生异常穿透节点
            raise LLMUnavailableError(
                f"JSON 任务三跳均失败：{light_error}；{deepseek_error}；{fallback_error}"
            ) from fallback_error


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
        return json.loads(text[start : end + 1])
    raise ValueError(f"响应中未找到合法 JSON: {raw[:100]}")
