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


def ensure_collection_loaded(client: MilvusClient, collection_name: str) -> None:
    """确保集合处于已加载状态（load 是查询/检索的前置条件）。

    Milvus 的集合需显式 load 到内存后才能 search/query，否则报
    ``collection not loaded``。幂等调用，由「创建集合」与「检索服务初始化」
    两处共同调用：前者覆盖首次建库，后者覆盖 Milvus 服务端重启导致集合被卸载。
    """
    try:
        client.load_collection(collection_name)
    except Exception as load_error:
        # 加载失败不阻断调用方：检索侧对异常有兜底（返回空结果集）
        print(f"[milvus_client] {collection_name} 加载失败: {load_error}")


def create_kb_collection() -> None:
    """创建通用知识库 Collection（幂等，并能修复「有集合无索引」的半成品状态）。

    索引选型说明：
    - dense 用 HNSW：图索引，高召回 + 低延迟，适合百万级以下校园知识库
      （M=16 / efConstruction=200 为官方推荐平衡参数；检索时 ef=64）
    - sparse 用倒排索引：稀疏向量只有倒排一种成熟索引形态

    幂等判断必须**同时校验集合与索引**：Milvus 建集合是「先建集合、再建索引」
    两步操作，中途失败会留下「有集合无索引」的半成品。若只判断集合是否存在，
    本函数会永久变成空操作，索引再也不会被创建——症状是数据能正常插入，
    但 load/search 报 ``index not found``，且静态审查与单元测试都发现不了。
    """
    client = get_milvus_client()

    # 索引参数用官方工厂构造：pymilvus 2.4 的 create_collection 期望 IndexParams 对象
    # （内部按列表迭代，每项须含 field_name），不能传「以字段名为键的 dict」——
    # 后者会让迭代拿到字符串键，报 'str' object has no attribute 'pop'
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    if client.has_collection(settings.MILVUS_COLLECTION):
        if not client.list_indexes(collection_name=settings.MILVUS_COLLECTION):
            client.create_index(
                collection_name=settings.MILVUS_COLLECTION,
                index_params=index_params,
            )
            print(f"[milvus_client] {settings.MILVUS_COLLECTION} 已存在但缺索引，已补建")
        ensure_collection_loaded(client, settings.MILVUS_COLLECTION)
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBED_DIM),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(
        fields=fields, description="小旦答通用知识库（dense+sparse 双路）"
    )

    client.create_collection(
        collection_name=settings.MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[milvus_client] 已创建 Collection: {settings.MILVUS_COLLECTION}")
    ensure_collection_loaded(client, settings.MILVUS_COLLECTION)


def create_faq_collection() -> None:
    """创建 FAQ 专用 Collection（幂等，含半成品修复，见 create_kb_collection）。

    FAQ 索引只需 dense 单路：FAQ 匹配是高置信度精确问答场景，
    语义向量足够，稀疏关键词路由收益有限。
    """
    client = get_milvus_client()

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="question_vec",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200},
    )

    if client.has_collection(settings.MILVUS_FAQ_COLLECTION):
        if not client.list_indexes(collection_name=settings.MILVUS_FAQ_COLLECTION):
            client.create_index(
                collection_name=settings.MILVUS_FAQ_COLLECTION,
                index_params=index_params,
            )
            print(f"[milvus_client] {settings.MILVUS_FAQ_COLLECTION} 已存在但缺索引，已补建")
        ensure_collection_loaded(client, settings.MILVUS_FAQ_COLLECTION)
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="similar_questions", dtype=DataType.JSON),
        FieldSchema(name="tags", dtype=DataType.JSON),
        FieldSchema(
            name="question_vec", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBED_DIM
        ),
    ]
    schema = CollectionSchema(fields=fields, description="小旦答 FAQ 高置信度专用索引")

    client.create_collection(
        collection_name=settings.MILVUS_FAQ_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[milvus_client] 已创建 Collection: {settings.MILVUS_FAQ_COLLECTION}")
    ensure_collection_loaded(client, settings.MILVUS_FAQ_COLLECTION)


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


def delete_chunks_by_source(source_file: str) -> int:
    """按来源文件删除既有 chunk（索引重建幂等：先删后插的基础操作）。

    来源标识是文档级元数据 ``metadata["source_file"]``（相对路径）。
    不能用 source_url 代替：手动上传的文档该字段一律为 ``manual_upload``，
    按它删除会把所有手动文档一起删掉。

    :param source_file: 相对路径形式的来源标识（由 corpus.load_document 写入）
    :return: 删除条数
    """
    client = get_milvus_client()
    deleted = client.delete(
        collection_name=settings.MILVUS_COLLECTION,
        filter=f'metadata["source_file"] == "{source_file}"',
    )
    return int(deleted.get("delete_count", 0))


def reset_collections() -> None:
    """删除两个 Collection（--rebuild 整库重建专用）。

    适用场景：切分策略 / 向量维度 / schema 变更；或既有索引早于
    ``source_file`` 字段引入——老 chunk 无法按来源定位，必须先整体清空，
    否则新旧数据并存且无法按来源增量替换。
    """
    client = get_milvus_client()
    for name in (settings.MILVUS_COLLECTION, settings.MILVUS_FAQ_COLLECTION):
        if client.has_collection(name):
            client.drop_collection(name)
            print(f"[milvus_client] 已删除 Collection: {name}")
