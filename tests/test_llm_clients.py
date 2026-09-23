"""LLM 客户端与三级降级链：终点兜底、连接复用、本地兜底开关、JSON 模式重试策略、思考模式开关。"""

import pytest

from config.settings import get_current_semester, settings


# ==================== 降级链终点兜底 ====================


def _boom(*args: object, **kwargs: object) -> str:
    """模拟端点完全不可达：任何调用都失败。"""
    raise ConnectionError("模拟端点不可达")


def test_terminal_fallback_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """降级链耗尽时抛可判定的 LLMUnavailableError，而非 openai 原生异常。

    这是 T27 的回归防线：此前最后一跳没有 try，原生异常会穿透节点打挂整轮请求。
    """
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    monkeypatch.setattr(llm, "_call_llm", _boom)

    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_qwen_json("测试")
    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_qwen("测试")


def test_intent_route_survives_llm_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """两端点同时故障：意图路由退回「简单问答」，不中断整轮。"""
    import importlib

    intent_module = importlib.import_module("agent.nodes.intent_route")
    llm = importlib.import_module("agent.llm_clients")

    def _outage(prompt: str, system: str | None = None) -> str:
        raise llm.LLMUnavailableError("两端点均不可用")

    monkeypatch.setattr(intent_module, "chat_qwen_json", _outage)

    result = intent_module.intent_route({"user_input": "选课什么时候截止"})
    assert result["intent"].category == "简单问答"
    assert result["intent"].rewritten_query == "选课什么时候截止"


def test_quality_check_passes_on_llm_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """质检组件故障时放行，不能反过来阻塞回答。"""
    import importlib

    quality_module = importlib.import_module("agent.nodes.quality_check")
    llm = importlib.import_module("agent.llm_clients")

    def _outage(prompt: str, system: str | None = None) -> str:
        raise llm.LLMUnavailableError("两端点均不可用")

    monkeypatch.setattr(quality_module, "chat_qwen_json", _outage)

    result = quality_module.quality_check(
        {
            "user_input": "选课什么时候截止",
            "retrieved_contexts": [],
            "generated_response": "9 月 15 日",
            "retry_count": 0,
        }
    )
    assert result["quality"].passed is True


def test_generate_returns_fallback_on_llm_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生成端点故障时返回兜底回复（尊重语言偏好），而非抛异常。"""
    import importlib

    from agent.nodes.generate import FALLBACK_EN, FALLBACK_ZH

    generate_module = importlib.import_module("agent.nodes.generate")
    monkeypatch.setattr(generate_module, "chat_deepseek", _boom)
    monkeypatch.setattr(generate_module, "fetch_recent_memories", lambda *a, **k: [])

    zh = generate_module.generate(
        {
            "user_input": "选课截止时间",
            "retrieved_contexts": [],
            "response_language": "中文",
        }
    )
    assert zh["generated_response"] == FALLBACK_ZH

    en = generate_module.generate(
        {
            "user_input": "deadline",
            "retrieved_contexts": [],
            "response_language": "English",
        }
    )
    assert en["generated_response"] == FALLBACK_EN


def test_chitchat_returns_fallback_on_llm_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """闲聊路径同样有兜底：低价值路径更不该中断整轮。"""
    import importlib

    from agent.nodes.chitchat_response import FALLBACK_ZH as CHITCHAT_ZH

    chitchat_module = importlib.import_module("agent.nodes.chitchat_response")
    monkeypatch.setattr(chitchat_module, "chat_deepseek", _boom)

    result = chitchat_module.chitchat_response({"user_input": "今天天气怎么样"})
    assert result["generated_response"] == CHITCHAT_ZH


def test_llm_clients_are_reused() -> None:
    """客户端对象必须复用（OpenAI 内部持有连接池，每次新建等于丢弃 keep-alive）。"""
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    assert llm._get_deepseek_client() is llm._get_deepseek_client()
    assert llm._get_light_llm_client() is llm._get_light_llm_client()


def test_local_fallback_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地兜底默认关闭：未启用时应抛 LLMUnavailableError，而不是静默返回空串。"""
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    monkeypatch.setattr(llm, "_call_llm", _boom)
    monkeypatch.setattr(settings, "FALLBACK_LLM_ENABLED", False)

    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_qwen("测试")


def test_local_fallback_takes_over_when_cloud_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """云端两跳全挂 + 本地兜底已启用 → 由本地模型接管轻量任务。

    这是「降级链也要有尽头」的最终一环：主用端点与 DeepSeek 都在云端，
    断网时若没有本地兜底，轻量任务只能整体失败。
    """
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    models_called: list[str] = []

    def _only_local_works(
        client, model, messages, temperature, json_mode=False, extra_body=None
    ):
        models_called.append(model)
        if model == settings.FALLBACK_LLM_MODEL:
            return '{"ok": true}'
        raise ConnectionError("模拟云端不可达")

    monkeypatch.setattr(llm, "_call_llm", _only_local_works)
    monkeypatch.setattr(settings, "FALLBACK_LLM_ENABLED", True)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "qwen2.5:7b-instruct")

    # 普通调用与 JSON 调用都应能落到本地兜底
    assert llm.chat_qwen("测试") == '{"ok": true}'
    assert llm.chat_qwen_json("测试") == '{"ok": true}'
    assert "qwen2.5:7b-instruct" in models_called


def test_generate_entry_does_not_use_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生成入口刻意不接本地小模型：云端全挂时应抛异常，由调用方给保守兜底文案。"""
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    monkeypatch.setattr(llm, "_call_llm", _boom)
    monkeypatch.setattr(settings, "FALLBACK_LLM_ENABLED", True)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "qwen2.5:7b-instruct")

    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_deepseek("测试")


def test_qwen_json_skips_retry_when_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端点不可达时跳过第二级重试，直接降级（否则要白等两个超时周期）。"""
    import importlib

    import openai

    llm = importlib.import_module("agent.llm_clients")
    models_called: list[str] = []

    def _tracked(
        client, model, messages, temperature, json_mode=False, extra_body=None
    ):
        models_called.append(model)
        raise openai.APIConnectionError(request=None)  # type: ignore[arg-type]

    monkeypatch.setattr(llm, "_call_llm", _tracked)
    # 钉住本地兜底开关：本用例只关心云端两跳的调用次数，
    # 若沿用 .env 里的 FALLBACK_LLM_ENABLED=true，降级链会多走一跳，
    # 断言把环境配置耦合进了测试（同一份代码在开发机与 CI 上结论不同）
    monkeypatch.setattr(settings, "FALLBACK_LLM_ENABLED", False)

    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_qwen_json("测试")

    # 只应发生两次调用：轻量端点一次 + DeepSeek 一次（第二级被跳过）
    assert models_called == [settings.LIGHT_LLM_MODEL, settings.DEEPSEEK_MODEL]


def test_qwen_json_still_retries_on_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非可达性错误（如端点不支持 response_format）仍应保留第二级重试。"""
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    models_called: list[str] = []

    def _tracked(
        client, model, messages, temperature, json_mode=False, extra_body=None
    ):
        models_called.append(model)
        raise ValueError("端点不支持 response_format")

    monkeypatch.setattr(llm, "_call_llm", _tracked)
    # 同上：钉住兜底开关，使断言只覆盖云端三跳
    monkeypatch.setattr(settings, "FALLBACK_LLM_ENABLED", False)

    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_qwen_json("测试")

    # 轻量端点两级 + DeepSeek 一级 = 三次
    assert models_called == [
        settings.LIGHT_LLM_MODEL,
        settings.LIGHT_LLM_MODEL,
        settings.DEEPSEEK_MODEL,
    ]


def test_thinking_switch_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """思考模式开关：默认关闭，且按模型族转成各自参数名并真的落到调用参数上。

    现行型号（DeepSeek-V4.1-Flash / Qwen3.8 系列）默认开启思考，代价是
    思考 token 按输出价计费，且厂商会改写 temperature（千问：小于 0.6 的值
    自动调整为 0.6）——本项目「轻量任务用 0.1 求确定性」的设计会被悄悄推翻。
    本用例把「配置 → 请求体片段 → 实际调用参数」这条链钉住，防止后人改参数名时静默失效。
    """
    import importlib

    llm = importlib.import_module("agent.llm_clients")
    captured: list[dict] = []

    def _capture(
        client, model, messages, temperature, json_mode=False, extra_body=None
    ) -> str:
        captured.append(extra_body or {})
        return "ok"

    monkeypatch.setattr(llm, "_call_llm", _capture)
    monkeypatch.setattr(settings, "DEEPSEEK_THINKING", False)
    monkeypatch.setattr(settings, "LIGHT_LLM_THINKING", False)

    llm.chat_qwen("测试")
    llm.chat_deepseek("测试")

    # 关闭思考：千问用扁平布尔，DeepSeek 用嵌套 type，两者参数名不同
    assert captured == [{"enable_thinking": False}, {"thinking": {"type": "disabled"}}]
