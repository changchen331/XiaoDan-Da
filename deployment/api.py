"""FastAPI 服务化入口：仅限校园网访问（由部署层网关 / 网络策略控制）。

启动方式（本地开发）：
    uvicorn deployment.api:app --host 0.0.0.0 --port 8080
Docker 环境由 docker-compose 编排启动（见 docker-compose.yml 的 xiaodan-api 服务）。

接口说明：
- POST /chat   对话主接口，完整走 Agent 流水线
                （情绪检测 → 意图路由 → 检索 → 生成 → 质检 → 记忆）
- GET  /health 健康检查（存活探测）

安全设计：
- user_id 一律使用调用方传入的内部 ID，服务端不采集姓名 / 学号
- 输入长度是**显式配置**的（`user_input` 的 `max_length=4000`，见 ChatRequest），
  而非依赖框架默认值——默认值随版本变化，安全边界不该跟着漂
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.graph import invoke

app = FastAPI(
    title="小旦答 API",
    description="复旦校园智能问答与心理健康监测系统",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    """对话请求体。

    conversation_history 与 session_id 二选一：
    - 同一 session_id 的请求自动共享 Checkpointer 短期记忆
    - 显式传入 history 用于跨端会话迁移（App 端本地缓存的上下文）
    """

    user_input: str = Field(
        ..., min_length=1, max_length=4000, description="用户原始输入"
    )
    user_id: str = Field("anonymous", description="用户内部 ID（脱敏标识）")
    session_id: str = Field("default", description="会话 ID")
    conversation_history: list = Field(default_factory=list, description="对话历史")
    user_profile: dict = Field(
        default_factory=lambda: {"role": "本科生"}, description="用户画像"
    )


class ChatResponse(BaseModel):
    """对话响应体：只暴露最终回答与情绪级别，内部流程细节不下发。"""

    session_id: str
    final_response: str
    emotion_level: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """对话主接口：一次请求完成情绪检测与校园问答的完整流水线。"""
    result = invoke(
        user_input=request.user_input,
        user_id=request.user_id,
        session_id=request.session_id,
        conversation_history=request.conversation_history,
        user_profile=request.user_profile,
    )
    emotion = result.get("emotion")
    return ChatResponse(
        session_id=request.session_id,
        final_response=result["final_response"],
        emotion_level=emotion.level if emotion is not None else "未知",
    )


@app.get("/health")
def health() -> dict:
    """存活探测：容器编排与负载均衡的健康检查端点。"""
    return {"status": "ok", "service": "xiaodan"}
