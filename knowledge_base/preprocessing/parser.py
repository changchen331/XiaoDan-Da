"""格式解析：Unstructured.io 统一处理 PDF / HTML / DOCX / TXT / Markdown。

解析层职责：将异构格式文件转换为统一的 Element 列表（标题 / 段落 / 表格），
保留文档结构信息供切分阶段使用。

解析策略选择：
- 'fast'：基于规则的快速解析（纯文本型 PDF 首选，速度快一个量级）
- 'hi_res'：基于视觉模型的精确布局分析（复杂版式 / 扫描件）
- 'auto'：按文件特征自动选择（默认）
"""
from unstructured.partition.auto import partition


def parse_document(file_path: str, strategy: str = "auto") -> list:
    """将原始文件解析为 Unstructured Element 列表。

    :param file_path: 文件路径（支持 .pdf / .html / .docx / .txt / .md）
    :param strategy: 解析策略（'fast' / 'hi_res' / 'auto'）
    :return: Element 列表，每个元素带类别标注（title / narrative_text / table 等），
        类别信息由 chunker 阶段用于识别切分边界
    """
    return partition(filename=file_path, strategy=strategy)


def elements_to_text(elements: list) -> str:
    """将 Element 列表拼接为纯文本。

    用空行分隔各 Element，保留段落边界（chunker 的递归切分依赖空行识别段落）。
    """
    return "\n\n".join(str(element) for element in elements)
