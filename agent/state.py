"""AgentState 及各节点结果模型：LangGraph 全图共享的唯一数据结构。

设计原则：
- 只存必要的中间结果，不存冗余数据，避免 State 随轮次膨胀
- 每个节点只读自己需要的字段、只写自己产出的字段，节点之间通过 State 解耦
- 全部字段 total=False：构造初始 State 时允许只传部分键（LangGraph 增量合并）
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class EmotionResult(BaseModel):
    """情绪检测结果（由模块二产出）。"""

    level: Literal["正常", "轻度困扰", "中度困扰", "高危"]
    source: str  # 判定来源："规则引擎" / "分类模型" / "规则+模型融合" / "语义判别"
    confidence: float  # 置信度 0-1
    # 危险表达的指向（本人/他人/引用/否定/无），由语义判别层给出；未判定时为 None。
    # 用途：上报记录区分"本人危机"与"代他人求助"——不把"室友活不下去"
    # 记成提问者本人的高危事件（见 emotion/semantic.py）
    referent: str | None = None


class IntentResult(BaseModel):
    """意图路由结果（由轻量模型产出）。"""

    category: Literal["简单问答", "复杂查询", "FAQ", "闲聊越界"]
    confidence: float
    rewritten_query: str  # 改写后的检索 query（口语 → 精准检索语句）


class QualityResult(BaseModel):
    """质量评估结果（由轻量模型产出）。"""

    passed: bool
    score: float  # 0-1
    reason: str  # 不合格时的原因描述（合格时为通过理由）


class RetrievedChunk(BaseModel):
    """单个检索结果：文本 + 元数据 + 相关性得分。"""

    text: str
    metadata: dict
    score: float


class AgentState(TypedDict, total=False):
    """LangGraph 全局状态。"""

    # ===== 输入 =====
    user_input: str  # 用户原始输入
    user_id: str  # 用户 ID（脱敏后的内部 ID，非真实姓名/学号）
    session_id: str  # 会话 ID（同时作为 Checkpointer 的 thread_id）
    conversation_history: list  # 最近 N 轮对话历史 [{"role", "content"}, ...]
    user_profile: dict  # 用户画像（本科生/研究生/留学生/教职工）

    # ===== 情绪检测结果 =====
    emotion: Optional[EmotionResult]

    # ===== 意图路由结果 =====
    intent: Optional[IntentResult]

    # ===== 语言偏好 =====
    # 用户期望的回复语言（"中文" / "English"）：由意图路由结合当前输入语言与
    # 对话中的明确语言指令（如"请用中文回答"）判定；经 Checkpointer 跨轮持久化，
    # 同一会话说过一次即持续生效。仅影响面向用户的输出（generate / care_suffix），
    # 不影响上报与高危关怀回复（固定中文，面向后台处理者）。
    response_language: str

    # ===== 检索结果 =====
    retrieved_contexts: list  # RetrievedChunk 字典列表（text + metadata + score）
    faq_hit: bool  # FAQ 高置信度命中标记（命中则跳过生成环节）

    # ===== 生成结果 =====
    generated_response: str  # 基于上下文生成的回答

    # ===== 质量评估结果 =====
    quality: Optional[QualityResult]
    retry_count: int  # 质量不合格时的重试计数

    # ===== 记忆相关 =====
    memory_summary: str  # 本轮对话摘要（写入长期记忆）

    # ===== 输出控制 =====
    final_response: str  # 最终输出给用户的回答
    should_end: bool  # 是否结束流程
