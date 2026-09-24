"""检索过滤表达式：按用户身份与学期时效构造 Milvus 过滤条件。

**为什么单独成一个模块**：它被两个检索节点共用（简单问答的 `retrieve`
与复杂查询的 `plan_and_retrieve`）。此前由后者 import 前者的内部符号，
节点之间形成横向依赖——而 `nodes/__init__.py` 的导入顺序决定了谁先被加载，
当前能跑通只是因为 `retrieve` 恰好不回引包级符号，一旦回引即成真环。
放在这里，两个节点平级引用同一份实现，导入顺序不再承载语义。
"""

from config.settings import get_current_semester


def build_filter_expr(user_profile: dict, include_audience: bool = True) -> str:
    """构造元数据过滤表达式：按用户身份与当前学期过滤知识库。

    身份过滤：文档 target_audience 为"全部"或与用户角色一致时可见
    时效过滤：文档 valid_semester 为"长期有效"或等于当前学期时可见
    （例如 2024 年的选课通知不应出现在 2026 年的检索结果中）

    :param user_profile: 用户画像，role 字段取值
        本科生 / 研究生 / 留学生 / 教职工；未知角色不做身份过滤
    :param include_audience: 是否保留身份过滤。质检重试时置 False（放宽召回）：
        答案可能恰好在受众不匹配的文档里；而时效过滤必须保留——
        把过期的选课通知交给模型，误导学生的代价高于答不上来
    """
    semester = get_current_semester()
    semester_condition = f'metadata["valid_semester"] in ["{semester}", "长期有效"]'
    if not include_audience:
        return semester_condition

    role = user_profile.get("role", "全部")
    # 未知角色（如教职工）放宽为全员可见，避免过滤条件过严导致空结果
    audiences = [role, "全部"] if role in ("本科生", "研究生", "留学生") else ["全部"]
    audience_list = ", ".join(f'"{audience}"' for audience in audiences)

    return f'metadata["target_audience"] in [{audience_list}] and {semester_condition}'
