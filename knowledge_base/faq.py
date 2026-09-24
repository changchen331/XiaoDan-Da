"""FAQ 专用索引：独立于通用向量库的高置信度精确问答。

设计动机：高频问题（如"校园卡丢了怎么办"）在通用 RAG 流程中每轮都要
经历 检索 → 生成 → 质量评估 三个环节，而其答案早已由管理员审定。
FAQ 专用索引让此类问题一步命中标准答案，兼顾响应速度与回答权威性。

数据结构（每条 FAQ）：
- question：标准问题
- answer：标准答案（管理员审核后录入）
- similar_questions：相似问法（与标准问题共享同一答案，扩大召回面）
- tags：主题标签

匹配策略：问题向量 top1 相似度 > 0.85（阈值可配）→ 直接返回标准答案，
不经过 LLM 生成；未命中自动降级到通用 RAG 流程。
"""

from config.settings import settings


class FAQIndex:
    """FAQ 专用 Collection 的构建与匹配。"""

    def __init__(self) -> None:
        # 重型导入放在**实例化处**：embeddings 会连带 FlagEmbedding / transformers
        # （实测约 11s），而"导入本模块"（Agent 启动、单测、脚本）并不需要它。
        # 顺带收窄依赖故障面——I1 的 torchvision 崩溃正是沿着这条链炸到导入方的。
        from knowledge_base.indexing.embeddings import get_embedder
        from knowledge_base.indexing.milvus_client import (
            ensure_collection_loaded,
            get_milvus_client,
        )

        self.client = get_milvus_client()
        self.embedder = get_embedder()
        # 同 RetrievalService：检索前置条件，服务初始化时统一保证
        ensure_collection_loaded(self.client, settings.MILVUS_FAQ_COLLECTION)

    def build(self, faq_entries: list) -> int:
        """从结构化 FAQ 数据构建独立索引（幂等：先清空既有行再写入）。

        向量化对象：标准问题 + 全部相似问法各自成行（共享同一 answer），
        一条 FAQ 的问法越多，被不同表述命中的概率越高。

        :param faq_entries: [{"question", "answer", "similar_questions", "tags"}, ...]
        :return: 实际写入的索引行数
        """
        # 收集所有需要向量化的问题文本（标准问题 + 相似问法）
        texts_to_encode: list = []
        for entry in faq_entries:
            texts_to_encode.append(entry["question"])
            texts_to_encode.extend(entry.get("similar_questions", []))

        vectors = self.embedder.encode(texts_to_encode)

        # 向量与 FAQ 行对齐：标准问题取首个向量，相似问法依次取后续向量
        rows: list = []
        cursor = 0
        for entry in faq_entries:
            question_count = 1 + len(entry.get("similar_questions", []))
            for vec in vectors[cursor : cursor + question_count]:
                rows.append(
                    {
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "similar_questions": entry.get("similar_questions", []),
                        "tags": entry.get("tags", []),
                        "question_vec": vec["dense"],
                    }
                )
            cursor += question_count

        # 幂等：先清空既有 FAQ 行再写入——重跑 build_index 不应累积重复问答。
        # FAQ 没有"按来源替换"可言（题库以 faq.json 为唯一来源），
        # 整体覆盖才是正确语义；id >= 0 匹配全部自增主键行
        self.client.delete(
            collection_name=settings.MILVUS_FAQ_COLLECTION, filter="id >= 0"
        )

        # 局部导入：与 __init__ 同一考虑，只在真正写库时才付加载代价
        from knowledge_base.indexing.milvus_client import insert_faq_rows

        insert_faq_rows(rows)
        return len(rows)

    def match(self, query: str, top_k: int = 3) -> dict | None:
        """高置信度匹配：最高相似度超过阈值返回标准答案，否则返回 None。

        :param query: 改写后的用户问题
        :param top_k: 召回的候选 FAQ 条数（取其最高分与阈值比较）
        :return: {"question", "answer", "score"} 或 None（未命中，调用方降级 RAG）
        """
        query_vec = self.embedder.encode([query])[0]

        results = self.client.search(
            collection_name=settings.MILVUS_FAQ_COLLECTION,
            data=[query_vec["dense"]],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            output_fields=["question", "answer"],
        )

        if results and results[0]:
            best = results[0][0]
            # IP 度量下 distance 即内积相似度（向量已归一化时取值 [0, 1]）
            score = float(best["distance"])
            if score > settings.FAQ_MATCH_THRESHOLD:
                return {
                    "question": best["entity"]["question"],
                    "answer": best["entity"]["answer"],
                    "score": score,
                }
        return None


# 服务级单例：Agent 的 FAQ 匹配节点直接导入使用
_faq_index: "FAQIndex | None" = None


def get_faq_index() -> FAQIndex:
    """获取全局 FAQ 索引实例（惰性初始化）。"""
    global _faq_index
    if _faq_index is None:
        _faq_index = FAQIndex()
    return _faq_index
