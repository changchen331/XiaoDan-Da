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
    # 密钥来源：优先项目变量名 DEEPSEEK_API_KEY，缺失时回退同名系统变量 DEEPSEEK
    # （便于把密钥只保存在操作系统环境里，不在 .env 中再存一份副本）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # 默认值刻意只写型号名、不带日期后缀：DeepSeek 侧型号更迭快
    # （旧名 deepseek-chat 已从官方模型列表移除），这里只保证「开箱可用」，
    # 具体型号以厂商文档为准，需要固定版本时由 .env 覆盖。
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    DEEPSEEK_TIMEOUT: int = int(
        os.getenv("DEEPSEEK_TIMEOUT", "30")
    )  # 生成环节 30 秒超时
    # 思考模式开关（默认关闭）。现行型号（DeepSeek-V4.1-Flash / Qwen3.8 系列）
    # **默认开启思考模式**，对本项目有两个副作用：
    # ① 思考 token 计入输出、按输出价计费——实测只回两个字也多耗 80+ token；
    # ② 思考模式下 temperature 会被厂商改写（千问文档：传入更小值自动调整为 0.6），
    #    而本项目轻量任务全部按 0.1 求确定性，等于设计意图被悄悄推翻。
    # 默认关闭：本项目的答案事实性由检索上下文保证，需要的是稳定可复现而非自由推理；
    # 需要开启时置 true 即可（厂商参数名不同，转换见 agent/llm_clients.py）。
    DEEPSEEK_THINKING: bool = (
        os.getenv("DEEPSEEK_THINKING", "false").lower() == "true"
    )

    # 轻量任务模型：意图路由 / 质量评估 / Query 改写 / 摘要 / FAQ 校验。
    #
    # 命名说明（重要）：这组变量**描述用途，不描述部署位置**。
    # 早先叫 LOCAL_LLM_*，但 base_url 实际指向云端兼容端点（百炼），
    # 名字与语义脱节——和"规则引擎输出'中度'、状态白名单却只认'中度困扰'"是同一类问题：
    # 名称一旦与实际不符，后人就会按名字去理解，从而误判架构。
    # 端点可指向本地推理服务，也可指向任意 OpenAI 兼容云端端点。
    # 注意：本组是**主用端点**，失败后的兜底见下方 FALLBACK_LLM_*。
    LIGHT_LLM_BASE_URL: str = os.getenv(
        "LIGHT_LLM_BASE_URL", "http://localhost:8000/v1"
    )
    # 密钥来源：优先 LIGHT_LLM_API_KEY，缺失时回退同名系统变量 QWEN（同上）
    LIGHT_LLM_API_KEY: str = os.getenv("LIGHT_LLM_API_KEY") or os.getenv(
        "QWEN", "dummy"
    )
    LIGHT_LLM_MODEL: str = os.getenv("LIGHT_LLM_MODEL", "qwen-plus")
    LIGHT_LLM_TIMEOUT: int = int(
        os.getenv("LIGHT_LLM_TIMEOUT", "15")
    )  # 轻量任务 15 秒超时
    # 思考模式开关（默认关闭，理由见上方 DEEPSEEK_THINKING）；
    # 对轻量任务而言关闭它是硬要求：意图路由 / 质检 / FAQ 校验要的是
    # 低温度下的确定性判定，而思考模式会把 temperature 强制改写为 0.6。
    LIGHT_LLM_THINKING: bool = (
        os.getenv("LIGHT_LLM_THINKING", "false").lower() == "true"
    )

    # 本地兜底模型：**仅当云端端点全部不可用时**接管轻量任务。
    #
    # 存在的意义是恢复"真正的离线可用性"——当前主用端点与降级端点（DeepSeek）
    # 都在云端，断网即全挂。模型选型待定，故默认关闭；
    # 选定后填入 FALLBACK_LLM_MODEL 并置 FALLBACK_LLM_ENABLED=true 即可生效。
    #
    # 本组**描述位置（本地）+ 角色（兜底）**，与 LIGHT_LLM_* 的命名维度不同，这是有意为之。
    FALLBACK_LLM_ENABLED: bool = (
        os.getenv("FALLBACK_LLM_ENABLED", "false").lower() == "true"
    )
    FALLBACK_LLM_BASE_URL: str = os.getenv(
        "FALLBACK_LLM_BASE_URL", "http://localhost:11434/v1"
    )  # 默认指向 Ollama 的 OpenAI 兼容端点
    FALLBACK_LLM_API_KEY: str = os.getenv("FALLBACK_LLM_API_KEY", "ollama")
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "")
    # 180 秒而非更短：本地量化模型**冷启动要先把权重读进内存**（实测 88s），
    # 超时给窄会让第一次调用必然失败——而它恰恰是"云端全挂"时唯一的出路。
    # 本值是同一默认值的唯一声明处，.env.example 与 docker-compose.yml 与之对齐
    FALLBACK_LLM_TIMEOUT: int = int(os.getenv("FALLBACK_LLM_TIMEOUT", "180"))

    # ==================== 模块一：知识库 ====================

    # Milvus 向量数据库
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "xiaodan_kb")  # 通用知识库
    MILVUS_FAQ_COLLECTION: str = os.getenv(
        "MILVUS_FAQ_COLLECTION", "xiaodan_faq"
    )  # FAQ 专用索引

    # BGE-M3 稠密向量维度（由模型结构决定，BGE-M3 输出固定 1024 维）
    EMBED_DIM: int = 1024
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

    # 检索参数：混合检索两路各召回 top20，重排后取 top5 送入生成
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    FAQ_MATCH_THRESHOLD: float = float(
        os.getenv("FAQ_MATCH_THRESHOLD", "0.85")
    )  # FAQ 高置信度命中阈值

    # HuggingFace 模型下载源（默认国内镜像；海外网络可改回 https://huggingface.co）
    # 注：真正生效靠 load_dotenv() 把 .env 写进进程环境变量——huggingface_hub
    # 在**导入时**读取该变量并固化，故导入顺序见 knowledge_base/indexing/embeddings.py
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

    # ==================== 模块二：情绪检测 ====================

    # 微调后的 XLM-RoBERTa 情绪分类模型存放路径（由 scripts/train_emotion_model.py 产出）
    EMOTION_MODEL_PATH: str = os.getenv("EMOTION_MODEL_PATH", "models/emotion-xlmr")
    EMOTION_LABELS: tuple = (
        "正常",
        "轻度困扰",
        "中度困扰",
        "高危",
    )  # 标签顺序即模型类别 id 顺序
    # 模型结论被采纳的门槛（两个用途，同一个值）：
    # ① 融合升级：规则判中度 + 模型高危概率超此值 → 升级为高危
    # ② 模型独立判定的总门槛：模型最高概率不超过此值时，其输出视为"四类近乎均匀、
    #    不构成证据"，一律以规则引擎为准。
    #    v2 实测（100 条语料训出的模型）：普通事务性提问的高危概率 0.25–0.30、
    #    真实高危样本 0.28–0.42，**两者几乎完全重叠**——即模型对高危没有区分度。
    #    若不加这道门槛，argmax 会把 95% 的普通提问判为高危，整条问答链路被劫持
    #    （实测 100 条评测问题有 96 条没走到检索）。
    # 因此本阈值当前的实际含义是：**模型侧暂不承担高危判定，安全网由规则引擎兜底**。
    # v3 语料扩到 1500 条后需重新标定该值（届时模型概率才会有区分度）。
    HIGH_RISK_PROB_THRESHOLD: float = float(
        os.getenv("HIGH_RISK_PROB_THRESHOLD", "0.5")
    )
    ESCALATION_ROUNDS: int = int(
        os.getenv("ESCALATION_ROUNDS", "3")
    )  # 连续 N 轮负面 → 升级一级

    # 语义判别层（规则快路径之后的 LLM 事实抽取，见 emotion/semantic.py）。
    # 为什么默认开启：规则引擎在隐晦表达上漏报严重（评测集 18 条高危命中 0 条），
    # 关掉它等于放弃这部分召回；而它失败时会**维持规则结论**，不会让系统更不安全。
    EMOTION_SEMANTIC_ENABLED: bool = (
        os.getenv("EMOTION_SEMANTIC_ENABLED", "true").lower() == "true"
    )
    # 边界带（中度困扰）追加采样次数：单次判级在「中度 ↔ 高危」之间会抖动，
    # 追加采样取最高等级以兑现"宁可误报不可漏报"。0 = 只判一次。
    EMOTION_SEMANTIC_ESCALATION: int = int(
        os.getenv("EMOTION_SEMANTIC_ESCALATION", "2")
    )

    # ==================== 模块二：高危上报 ====================

    ALERT_EMAIL_ENABLED: bool = (
        os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    )
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
    RECURSION_LIMIT: int = int(
        os.getenv("RECURSION_LIMIT", "25")
    )  # LangGraph 递归上限，兜底防死循环
    HISTORY_MAX_ROUNDS: int = int(
        os.getenv("HISTORY_MAX_ROUNDS", "10")
    )  # 生成时携带的最大历史轮数

    # ==================== 模块四：评估与可观测 ====================

    # RAGAS 裁判模型（LLM-as-a-Judge）：与生成模型（DeepSeek）跨族去偏，
    # 且**必须用带日期后缀的快照版本**——模型静默升级会让 v2/v3 的评测分数不可比，
    # 而纵向可复现性正是这套离线评测存在的意义。
    #
    # 选型理由（在百炼 255 个模型中的取舍）：
    # - 不选非 Qwen 的第三方族（GLM / Kimi）：经 API 实查，只有 Qwen 提供日期快照；
    #   Kimi 另有硬伤——temperature 只接受 1，与「裁判必须 temperature=0」冲突
    # - 不选 Qwen 旗舰 qwen3.7-max：实测 6.6s vs 1.7s，100 条 × 5 指标差一个数量级耗时
    # - 因此选 qwen3-max-2026-01-23：快照锁定 + 实测最快 + 与在线质检同族（口径对齐）
    # - 残余的自我偏好风险有限：主答案为 DeepSeek 生成，裁判 Qwen 对它并不自评
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "qwen3-max-2026-01-23")
    JUDGE_BASE_URL: str = os.getenv(
        "JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 裁判密钥：优先 JUDGE_API_KEY，缺失时回退系统变量 QWEN
    # （与上方 DEEPSEEK/LIGHT_LLM 的「优先项目变量名、再回退系统变量」读取顺序一致）
    JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY") or os.getenv("QWEN", "")

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
