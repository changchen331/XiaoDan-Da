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

# 先装 CPU 版 torch 再装其余依赖（pip 检测到已满足会自动跳过）：
# CPU 镜像体积约为 CUDA 版的 1/6，demo 场景推理延迟完全可接受；
# GPU 训练在宿主机 conda 环境进行，不进容器
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖（利用层缓存：依赖未变时重构建秒级完成）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（按模块分组拷贝，data/ 与 models/ 通过 volume 挂载注入）
COPY config/ config/
COPY agent/ agent/
COPY knowledge_base/ knowledge_base/
COPY emotion/ emotion/
COPY evaluation/ evaluation/
COPY observability/ observability/
COPY deployment/ deployment/

# HuggingFace 模型缓存目录：BGE-M3 / Reranker 首次启动自动下载后持久化
ENV HF_HOME=/app/hf_cache
# 国内网络加速镜像源（可在运行时覆盖为官方源）
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8080

# 仅启动 API 服务（Milvus / PostgreSQL / Langfuse 由 docker-compose 提供）
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8080"]
