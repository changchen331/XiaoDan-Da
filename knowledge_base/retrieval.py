"""检索服务：混合检索 + 重排的完整流水线。

检索五步流程（对应架构文档模块一）：
1. Query 改写 —— 在 Agent 意图路由节点完成（口语 → 精准检索语句）
2. 向量化 —— BGE-M3 一次编码同时产出 dense + sparse
3. 元数据过滤 —— 按用户身份（本科生/研究生/…）与学期时效性过滤
4. 混合检索 —— Milvus 两路各召回 top20，RRF（Reciprocal Rank Fusion）融合排序
5. 重排 —— BGE-Reranker-v2-M3 对融合候选做交叉编码精排，取 top5 送入生成

RRF 融合公式：score(d) = Σ 1/(k + rank_i(d))，k=60（标准值），
不依赖两路得分的绝对量纲，天然适合 dense/sparse 异构分数融合。
"""
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from config.settings import settings
from knowledge_base.indexing.embeddings import get_embedder
from knowledge_base.indexing.milvus_client import get_milvus_client

# 进程级单例：Reranker 模型体积大，与 Embedder 一样只加载一次
_reranker = None


class RetrievalService:
    """混合检索 + 重排服务（通过 get_retrieval_service() 获取全局单例）。"""

    def __init__(self) -> None:
        self.client: MilvusClient = get_milvus_client()
        self.embedder = get_embedder()

    def hybrid_search(self, query: str, filter_expr: str = "", top_k: int | None = None) -> list:
        """混合检索：dense（语义）+ sparse（关键词）两路召回后 RRF 融合。

        :param query: 改写后的检索语句
        :param filter_expr: 元数据过滤表达式（Milvus JSON 路径语法），如：
            'metadata["target_audience"] in ["本科生", "全部"] and
             metadata["valid_semester"] in ["2026-2027秋季", "长期有效"]'
            为空时不做元数据过滤（全库检索）。
        :param top_k: 单路召回数量，默认取 settings.HYBRID_TOP_K（20）
        :return: [{"text", "metadata", "score"}, ...] 按 RRF 融合分数降序
        """
        limit = top_k or settings.HYBRID_TOP_K

        # query 侧编码：同时拿到 dense 与 sparse 两路查询向量
        query_vec = self.embedder.encode([query], is_query=True)[0]

        # 构造两路检索请求：dense 走 HNSW，sparse 走倒排索引，度量均为 IP（内积）
        dense_request = AnnSearchRequest(
            data=[query_vec["dense"]],
            anns_field="dense",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=limit,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[query_vec["sparse"]],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=limit,
            expr=filter_expr,
        )

        results = self.client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(k=60),  # k=60 为 RRF 论文与工业实践的标准平滑常数
            limit=limit,
            output_fields=["text", "metadata"],
        )
        return self._format_results(results)

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """BGE-Reranker-v2-M3 交叉编码重排，取 top_k 返回给 Agent。

        Reranker 与向量检索的区别：向量检索是 query 和文档各自独立编码后算相似度
        （双塔结构），Reranker 将 query 与文档拼接后联合编码（交叉注意力），
        能捕捉细粒度的语义关联，精度显著更高但计算量更大，
        因此只用于对召回候选的精排阶段。

        :param query: 原始（改写后）查询
        :param candidates: hybrid_search 的返回值
        :param top_k: 精排后保留的片段数
        :return: 按相关性降序的候选子集
        """
        if not candidates:
            return []

        reranker = _get_reranker()
        pairs = [[query, candidate["text"]] for candidate in candidates]

        # compute_score 返回每个 (query, doc) pair 的相关性分数
        scores = reranker.compute_score(pairs, normalize=True)

        # 按重排分数降序取 top_k（scores 为浮点列表时逐项对应）
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        reranked = []
        for candidate, score in scored[:top_k]:
            reranked.append({
                "text": candidate["text"],
                "metadata": candidate["metadata"],
                "score": float(score),  # 覆盖召回分数，保留重排分数
            })
        return reranked

    @staticmethod
    def _format_results(raw_results: list) -> list:
        """将 Milvus hybrid_search 的原始返回整理为统一结构。

        hybrid_search 对单条 query 返回 [[Hit, Hit, ...]]，
        每个 Hit 含 id / distance / entity（entity 即 output_fields 的值）。
        """
        formatted: list = []
        for hits in raw_results:
            for hit in hits:
                formatted.append({
                    "text": hit["entity"]["text"],
                    "metadata": hit["entity"]["metadata"],
                    "score": float(hit["distance"]),
                })
        return formatted


def _get_reranker():
    """获取全局唯一的 Reranker 实例（惰性初始化）。"""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(settings.RERANK_MODEL_NAME, use_fp16=True)
    return _reranker


# 服务级单例：Agent 检索节点直接导入使用
_retrieval_service: "RetrievalService | None" = None


def get_retrieval_service() -> RetrievalService:
    """获取全局检索服务实例（惰性初始化，模型只加载一次）。"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
