# ============================================================
# 小旦答 API 服务镜像
# 构建：docker build -t xiaodan-api -f deployment/Dockerfile .
# 运行：由 docker-compose.yml 统一编排（Milvus / PostgreSQL 同栈启动）
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# 系统依赖：
# - libgomp1      : PyTorch / BGE 模型推理所需的 OpenMP 运行时
# - poppler-utils : Unstructured 解析 PDF 所需的 poppler 工具集
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# uv 包管理器（从官方镜像直接拷贝二进制，无需 pip 安装）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 虚拟环境固定位置；copy 模式适配容器内无硬链接的场景
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_LINK_MODE=copy

# 按锁定文件精确安装依赖（--frozen：不尝试重新解析，与 uv.lock 严格一致）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# torch 换为 CPU 版：镜像体积约为 CUDA 版的 1/6，demo 场景推理延迟完全可接受；
# GPU 训练在宿主机进行，不进容器。此步与 lock 的差异为刻意设计
# （uv pip 为自由安装，不受 lock 约束回滚）
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    --python /app/.venv/bin/python

# 复制项目代码（按模块分组拷贝，data/ 与 models/ 通过 volume 挂载注入）
COPY config/ config/
COPY infra/ infra/
COPY agent/ agent/
COPY knowledge_base/ knowledge_base/
COPY emotion/ emotion/
COPY evaluation/ evaluation/
COPY observability/ observability/
COPY deployment/ deployment/
COPY scripts/ scripts/

# HuggingFace 模型缓存目录：BGE-M3 / Reranker 首次启动自动下载后持久化
ENV HF_HOME=/app/hf_cache
# 国内网络加速镜像源（可在运行时覆盖为官方源）
ENV HF_ENDPOINT=https://hf-mirror.com

# 虚拟环境进入 PATH：CMD 直接使用其中的 uvicorn / python
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

# 仅启动 API 服务（Milvus / PostgreSQL / Langfuse 由 docker-compose 提供）
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8080"]
