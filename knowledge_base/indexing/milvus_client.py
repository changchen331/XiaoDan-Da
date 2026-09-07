"""Milvus 集合管理：通用知识库与 FAQ 专用索引的建表与写入。

通用知识库 Collection（xiaodan_kb）字段设计：
- id：自增主键
- text：chunk 原文（生成环节的唯一事实来源）
- dense：BGE-M3 稠密向量（1024 维，HNSW 索引，IP 内积度量）
- sparse：BGE-M3 稀疏向量（倒排索引，IP 度量）
- metadata：结构化元数据 JSON（source_url / doc_type / target_audience /
  valid_semester / publish_date / language / topic_tag）

FAQ 专用 Collection（xiaodan_faq）字段设计：
- id：自增主键
- question：标准问题原文
- answer：标准答案（高置信度命中时直接返回，不经过 LLM 改写，保证权威性）
- similar_questions：相似问法列表（JSON 数组，与标准问题共享同一答案）
- tags：主题标签（JSON 数组，用于管理后台分类展示）
- question_vec：问题向量（dense，HNSW 索引）
"""
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from config.settings import settings


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端（MilvusClient 内部维护连接池，可重复创建）。"""
    return MilvusClient(uri=settings.MILVUS_URI)


def create_kb_collection() -> None:
    """创建通用知识库 Collection（幂等：已存在则直接返回）。

    索引选型说明：
    - dense 用 HNSW：图索引，高召回 + 低延迟，适合百万级以下校园知识库
      （M=16 / efConstruction=200 为官方推荐平衡参数；检索时 ef=64）
    - sparse 用倒排索引：稀疏向量只有倒排一种成熟索引形态
    """
    client = get_milvus_client()
    if client.has_collection(settings.MILVUS_COLLECTION):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBED_DIM),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields=fields, description="小旦答通用知识库（dense+sparse 双路）")

    client.create_collection(
        collection_name=settings.MILVUS_COLLECTION,
        schema=schema,
        index_params={
            "dense": {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200},
            },
            "sparse": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        },
    )
    print(f"[milvus_client] 已创建 Collection: {settings.MILVUS_COLLECTION}")


def create_faq_collection() -> None:
    """创建 FAQ 专用 Collection（幂等）。

    FAQ 索引只需 dense 单路：FAQ 匹配是高置信度精确问答场景，
    语义向量足够，稀疏关键词路由收益有限。
    """
    client = get_milvus_client()
    if client.has_collection(settings.MILVUS_FAQ_COLLECTION):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="similar_questions", dtype=DataType.JSON),
        FieldSchema(name="tags", dtype=DataType.JSON),
        FieldSchema(name="question_vec", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="小旦答 FAQ 高置信度专用索引")

    client.create_collection(
        collection_name=settings.MILVUS_FAQ_COLLECTION,
        schema=schema,
        index_params={
            "question_vec": {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200},
            },
        },
    )
    print(f"[milvus_client] 已创建 Collection: {settings.MILVUS_FAQ_COLLECTION}")


def insert_chunks(chunks: list, vectors: list) -> int:
    """将 chunk 与对应向量批量写入通用知识库。

    :param chunks: chunk 列表，每项为 {"text": str, "metadata": dict}
    :param vectors: get_embedder().encode() 的输出，每项为 {"dense": ..., "sparse": ...}
    :return: 实际写入条数
    """
    client = get_milvus_client()

    rows = [
        {
            "text": chunk["text"],
            "dense": vec["dense"],
            "sparse": vec["sparse"],
            "metadata": chunk["metadata"],
        }
        for chunk, vec in zip(chunks, vectors)
    ]
    client.insert(collection_name=settings.MILVUS_COLLECTION, data=rows)
    return len(rows)


def insert_faq_rows(rows: list) -> int:
    """将 FAQ 行批量写入专用 Collection。

    :param rows: 每项为 {"question", "answer", "similar_questions", "tags", "question_vec"}
    :return: 实际写入条数
    """
    client = get_milvus_client()
    client.insert(collection_name=settings.MILVUS_FAQ_COLLECTION, data=rows)
    return len(rows)
