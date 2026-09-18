"""文档处理 - 文本清洗：去除页眉页脚、导航栏、广告等噪声信息。"""

import re

# 不可见字符（零宽空格、BOM、行分隔符等）：官网条目文本与正文中实测存在，
# 会切断分词、污染文件名与元数据标题，统一在清洗阶段剔除
INVISIBLE_CHAR_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]")

# 常见噪声模式：页眉页脚 / 版权声明 / 网站导航残留 / 网文元信息
NOISE_PATTERNS: tuple = (
    r"^复旦大学\s*(首页|官网).*$",  # 导航栏残留
    r"^版权所有.*$|^©.*$|^Copyright.*$|^All Rights Reserved.*$",  # 版权声明
    r"^第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$",  # 页码
    r"^(ICP备案|沪ICP).*号?$",  # 备案号
    r"^分享到.*(微信|微博|QQ).*$",  # 分享按钮残留
    r"^(发布者|作者|来源|编辑|审核|供稿)[:：].*$",  # 网文署名元信息
    r"^发布时间[:：].*$",  # 网文发布时间（发布日期由爬虫单独写入元数据头部）
)

# 跨行噪声：浏览量标签与其后的数字常分作两行（"浏览次数：" 换行 "10"），
# 逐行规则无法覆盖，需在切行之前整体替换。
# 注意标签后的空白用 [ \t] 而非 \s：\s 会把换行一并吃掉，导致紧随其后的
# 数字失去匹配对象而残留在正文里
INLINE_NOISE_PATTERNS: tuple = (
    r"浏览次数[:：]?[ \t]*(?:\n\s*\d+)?",
    r"点击次数[:：]?[ \t]*(?:\n\s*\d+)?",
)


def clean_text(text: str) -> str:
    """逐行过滤噪声 + 规整段落边界，避免噪声内容干扰后续检索和生成。

    空行必须保留：parser.elements_to_text 以空行分隔各 Element，手册类切分的
    递归策略也按空行（"\\n\\n"）识别段落边界——若把空行全部删除，
    文档段落结构会被压平，切分只能退化到按单换行甚至句号切。
    因此这里只把连续多个空行压缩为一个，而非删除。
    """
    for inline_pattern in INLINE_NOISE_PATTERNS:
        text = re.sub(inline_pattern, "", text)
    text = INVISIBLE_CHAR_RE.sub("", text)

    kept_lines: list = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            if not any(re.match(pattern, stripped) for pattern in NOISE_PATTERNS):
                kept_lines.append(stripped)
        elif kept_lines and kept_lines[-1] != "":
            kept_lines.append("")  # 段落边界：连续空行压缩为一个

    while kept_lines and kept_lines[-1] == "":
        kept_lines.pop()  # 去掉首尾多余空行
    return "\n".join(kept_lines)
