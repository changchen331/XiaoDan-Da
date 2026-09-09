"""全局配置中心：集中管理所有环境变量与运行参数。

配置加载优先级：``.env`` 文件（本地开发）→ 系统环境变量（Docker 容器）→ 代码默认值。
全项目统一通过 ``from config.settings import settings`` 获取配置，
不允许出现散落的 ``os.getenv`` 调用，保证配置可审计、可覆盖、可脱敏。

敏感信息（API Key、数据库密码、邮箱凭证）一律不入库，
仅在 ``.env``（已被 .gitignore 屏蔽）或容器环境变量中维护。
"""
import os
from datetime import date

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件；文件不存在时静默跳过（Docker 环境直接读环境变量）
load_dotenv()


class Settings:
    """集中管理所有配置项。环境变量优先，缺省值面向本地开发环境。"""

    # ==================== 模块三：LLM 服务 ====================

    # DeepSeek-V3 API：回答生成、关怀回复等核心质量环节
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_TIMEOUT: int = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))  # 生成环节 30 秒超时

    # 轻量任务模型（Qwen2.5-14B）：意图路由 / 质量评估 / Query 改写 / 摘要。
    # base_url 可指向本地 vLLM，也可指向任意 OpenAI 兼容的云端端点（如 SiliconFlow），
    # 端点不可用时轻量任务自动降级到 DeepSeek，保证任何环境都能完整跑通流程。
    LOCAL_LLM_BASE_URL: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    LOCAL_LLM_API_KEY: str = os.getenv("LOCAL_LLM_API_KEY", "dummy")
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
    LOCAL_LLM_TIMEOUT: int = int(os.getenv("LOCAL_LLM_TIMEOUT", "15"))  # 轻量任务 15 秒超时

    # ==================== 模块一：知识库 ====================

    # Milvus 向量数据库
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "xiaodan_kb")  # 通用知识库
    MILVUS_FAQ_COLLECTION: str = os.getenv("MILVUS_FAQ_COLLECTION", "xiaodan_faq")  # FAQ 专用索引

    # BGE-M3 稠密向量维度（由模型结构决定，BGE-M3 输出固定 1024 维）
    EMBED_DIM: int = 1024
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

    # 检索参数：混合检索两路各召回 top20，重排后取 top5 送入生成
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    FAQ_MATCH_THRESHOLD: float = float(os.getenv("FAQ_MATCH_THRESHOLD", "0.85"))  # FAQ 高置信度命中阈值

    # HuggingFace 模型下载源（国内网络可切换为 https://hf-mirror.com 加速）
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://huggingface.co")

    # ==================== 模块二：情绪检测 ====================

    # 微调后的 XLM-RoBERTa 情绪分类模型存放路径（由 scripts/train_emotion_model.py 产出）
    EMOTION_MODEL_PATH: str = os.getenv("EMOTION_MODEL_PATH", "models/emotion-xlmr")
    EMOTION_LABELS: tuple = ("正常", "轻度困扰", "中度困扰", "高危")  # 标签顺序即模型类别 id 顺序
    HIGH_RISK_PROB_THRESHOLD: float = float(os.getenv("HIGH_RISK_PROB_THRESHOLD", "0.5"))
    ESCALATION_ROUNDS: int = int(os.getenv("ESCALATION_ROUNDS", "3"))  # 连续 N 轮负面 → 升级一级

    # ==================== 模块二：高危上报 ====================

    ALERT_EMAIL_ENABLED: bool = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    ALERT_SMTP_HOST: str = os.getenv("ALERT_SMTP_HOST", "")
    ALERT_SMTP_PORT: int = int(os.getenv("ALERT_SMTP_PORT", "465"))
    ALERT_SENDER: str = os.getenv("ALERT_SENDER", "")
    ALERT_RECEIVER: str = os.getenv("ALERT_RECEIVER", "")
    ALERT_SMTP_PASSWORD: str = os.getenv("ALERT_SMTP_PASSWORD", "")

    # ==================== 模块二：轻/中度关怀渠道信息 ====================

    # 心理咨询中心的求助渠道（中英各一份），供 care_suffix 节点在中度困扰的
    # 回复末尾附加。从 .env 注入而非写死在代码 / LLM prompt 里：
    # - 脱敏：仓库内只留占位符，公开仓库不携带真实联系方式
    # - 防编造：prompt 要求 LLM 原样保留渠道文本，电话 / 链接绝不出自模型
    CARE_CENTER_INFO_ZH: str = os.getenv(
        "CARE_CENTER_INFO_ZH",
        "如需支持，可联系学校心理咨询中心（联系方式请向辅导员或校官网查询）。",
    )
    CARE_CENTER_INFO_EN: str = os.getenv(
        "CARE_CENTER_INFO_EN",
        "If you need support, please contact the university counseling center "
        "(contact details available from your counselor or the official website).",
    )

    # ==================== 模块三：流程控制 ====================

    MAX_RETRY: int = int(os.getenv("MAX_RETRY", "2"))  # 质量评估不合格最多重试 2 次
    RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "25"))  # LangGraph 递归上限，兜底防死循环
    HISTORY_MAX_ROUNDS: int = int(os.getenv("HISTORY_MAX_ROUNDS", "10"))  # 生成时携带的最大历史轮数

    # ==================== 模块四：评估与可观测 ====================

    # RAGAS 裁判模型：GPT-4o 快照版本（版本锁定保证分数跨迭代可比），与生成模型跨族去偏
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o-2024-08-06")

    # Langfuse 全链路追踪（开关式）：公钥私钥都配置时启用，否则所有追踪函数为空操作
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # ==================== 数据库 ====================

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "xiaodan")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "changeme")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "xiaodan")

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL 连接串（Checkpointer 短期记忆 / user_memory 长期记忆 / emotion_alerts 上报表共用实例）"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def langfuse_enabled(self) -> bool:
        """Langfuse 是否启用：公钥与私钥均已配置时才开启。"""
        return bool(self.LANGFUSE_PUBLIC_KEY) and bool(self.LANGFUSE_SECRET_KEY)


def get_current_semester(today: date | None = None) -> str:
    """计算当前复旦学期标识，用于知识库的时效性过滤。

    学期划分规则（与校历一致）：
    - 9 月至次年 1 月为秋季学期
    - 2 月至 6 月为春季学期
    - 7 月至 8 月为暑期学期

    :param today: 基准日期，默认取今天（注入参数便于测试）
    :return: 形如 "2026-2027秋季" 的学期字符串
    """
    current = today or date.today()
    year, month = current.year, current.month
    if month >= 9:
        return f"{year}-{year + 1}秋季"
    if month <= 1:
        return f"{year - 1}-{year}秋季"
    if month <= 6:
        return f"{year - 1}-{year}春季"
    return f"{year - 1}-{year}暑期"


settings = Settings()
