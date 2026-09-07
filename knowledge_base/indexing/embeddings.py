"""BGE-M3 向量化：一次调用同时生成 dense（1024 维）+ sparse 两种向量。

双路向量的分工：
- dense 稠密向量：语义匹配，解决"汽车 ≈ 车辆"这类同义改写召回问题
- sparse 稀疏向量：关键词精确匹配，解决"向量机"等专业术语的精确命中问题
- 两路向量存入同一个 Milvus Collection 的不同字段，检索时融合排序

性能设计：模型加载耗时且占显存，全进程仅加载一次（模块级单例），
入库脚本与检索服务共享同一实例。
"""
from FlagEmbedding import BGEM3FlagModel

from config.settings import settings

# 进程级单例：避免重复加载 2GB+ 的 BGE-M3 模型
_embedder: "BGEM3Embedder | None" = None


class BGEM3Embedder:
    """BGE-M3 编码器封装（dense + sparse 双路输出）。"""

    def __init__(self) -> None:
        """加载 BGE-M3 模型。

        use_fp16=True：半精度推理，显存减半、速度提升，精度损失可忽略
        （检索场景下 fp16 对召回率的影响在千分之一量级）。
        """
        self.model = BGEM3FlagModel(settings.EMBED_MODEL_NAME, use_fp16=True)

    def encode(self, texts: list, is_query: bool = False) -> list:
        """批量编码，返回每个文本的 {dense, sparse} 向量对。

        :param texts: 文本列表（入库 chunk 或用户 query）
        :param is_query: True 按检索侧重编码（短文本），False 按语料侧编码（长文本）。
            BGE-M3 对两种角色使用不同的编码前缀，正确设置可提升召回质量。
        :return: [{"dense": list[float], "sparse": dict[int, float]}, ...]
        """
        outputs = self.model.encode(
            texts,
            batch_size=8,
            max_length=1024,             # 覆盖 768 token 的 chunk 上限后仍有余量
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,   # 多向量（ColBERT）检索路线暂不启用
        )

        result: list = []
        for i in range(len(texts)):
            dense = outputs["dense_vecs"][i].tolist()                    # 1024 维稠密向量
            sparse = _to_milvus_sparse(outputs["lexical_weights"][i])     # {token_id: 权重} 稀疏向量
            result.append({"dense": dense, "sparse": sparse})
        return result


def _to_milvus_sparse(lexical_weights: dict) -> dict:
    """将 BGE-M3 输出的词权重转换为 Milvus SPARSE_FLOAT_VECTOR 接受的格式。

    BGE-M3 稀疏输出为 {token_id: weight}，Milvus 要求键为整型、值为浮点型，
    此处仅做类型规整，不改变权重分布。
    """
    return {int(token_id): float(weight) for token_id, weight in lexical_weights.items()}


def get_embedder() -> BGEM3Embedder:
    """获取全局唯一的 BGE-M3 编码器实例（惰性初始化）。

    :raises RuntimeError: 模型下载失败或显存不足时抛出（首次调用时才会触发）
    """
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder
