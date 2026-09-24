"""FAQ 命中的轻量校验：防止"向量相似但语义不同"的错误命中直返给用户。

为什么 0.85 的高阈值还需要校验：
- 向量相似度衡量的是表述接近，不是答案适配。"校园卡丢了怎么办"与
  "校园卡补办收费吗"可能高度相似，但标准答案答非所问
- 中文 FAQ 库 + 英文用户：即使语义匹配，直接返回中文答案也不符合
  用户语言期望

校验失败的处理策略（信任阈值）：任何异常（端点不可达 / JSON 解析失败）
时返回 True（视为校验通过）。校验是增强层，增强层故障不应拖垮
0.85 阈值本身的基线判别能力——错误命中是低概率事件，而"FAQ 永远
无法命中"是确定性故障，两害相权取其轻。
"""

from infra.llm_clients import JSON_TASK_ERRORS, chat_qwen_json_parsed

VERIFY_PROMPT = """请校验一条 FAQ 匹配结果是否可以安全地直接返回给用户，检查两个维度：

1. answer_match：标准答案是否确实回答了用户的问题？
   （警惕"问 A 答 B"：两者话题相近但侧重点不同时判 false）
2. language_match：标准答案的语言是否与用户期望的回复语言一致？

用户问题：{query}
用户期望的回复语言：{language}
FAQ 标准问题：{question}
FAQ 标准答案：{answer}

请以JSON格式输出：
{{"answer_match": true, "language_match": true}}"""


def verify_faq_match(
    query: str, question: str, answer: str, response_language: str
) -> bool:
    """校验 FAQ 命中结果（答案适配 + 语言匹配），两维全过才放行。

    :param query: 用户改写后的检索 query
    :param question: FAQ 索引返回的标准问题
    :param answer: FAQ 索引返回的标准答案
    :param response_language: 用户期望的回复语言（中文 / English）
    :return: True 表示可直返标准答案；False 表示应视同未命中走通用检索。
        校验调用自身失败时返回 True（信任阈值，见模块说明）
    """
    try:
        result = chat_qwen_json_parsed(
            VERIFY_PROMPT.format(
                query=query,
                language=response_language,
                question=question,
                answer=answer,
            )
        )
        return bool(result["answer_match"]) and bool(result["language_match"])
    except JSON_TASK_ERRORS as verify_error:
        # 信任阈值策略：校验层故障时放行（低概率错误命中 vs 确定性快路径失效）
        print(f"[faq_verify] 校验调用失败，信任阈值直接放行: {verify_error}")
        return True
