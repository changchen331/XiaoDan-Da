"""文档处理 - 文本清洗：去除页眉页脚、导航栏、广告等噪声信息。"""
import re

# 常见噪声模式：页眉页脚 / 版权声明 / 网站导航残留
NOISE_PATTERNS: tuple = (
    r"^复旦大学\s*(首页|官网).*$",               # 导航栏残留
    r"^版权所有.*$|^©.*$|^Copyright.*$|^All Rights Reserved.*$",  # 版权声明
    r"^第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$",   # 页码
    r"^(ICP备案|沪ICP).*号?$",                  # 备案号
    r"^分享到.*(微信|微博|QQ).*$",               # 分享按钮残留
)


def clean_text(text: str) -> str:
    """逐行过滤噪声 + 合并多余空行，避免噪声内容干扰后续检索和生成。"""
    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped) for p in NOISE_PATTERNS):
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines)
