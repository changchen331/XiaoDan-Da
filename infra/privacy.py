"""出域数据最小化：跨模块共用的脱敏口径（**唯一定义处**）。

系统的数据出口只有两条，此前各写一套口径、互不知情：

1. **高危上报**（`emotion/reporting.py`）——触发文本摘要写入 `emotion_alerts` 表，
   并随告警邮件发给心理咨询中心值班邮箱；
2. **离线评测**（`evaluation/ragas_eval.py`）——评测集（含真实学生提问）送给外部裁判模型。

改造前：上报侧只做"截断前 100 字"、**不做正则脱敏**，手机号 / 学号会原样落库并进邮件；
评测侧做了正则脱敏却不截断。两边都答不出"哪类数据算已脱敏"。
现在两侧都经 :func:`sanitize_outbound`——"什么算出域、出域前必须做什么"只在这里定义一次。

**顺序：先脱敏、再截断**。反过来则"截断处正好切在号码中间"时会留下半截数字，
既躲过正则、又仍属可识别的个人信息；宁可出现被截断的占位符，也不放半截号码出域。
"""

import re

#: 个人信息模式 → 占位符。**手机号必须排在前面**：它本身也是 8–11 位数字，
#: 若先被学号模式吃掉，报告里就分不清"这是手机号还是学号"（两者的处置口径不同）。
_PII_PATTERNS: tuple = (
    (re.compile(r"1[3-9]\d{9}"), "[手机号]"),
    (re.compile(r"\b\d{8,11}\b"), "[学号]"),
)


def mask_pii(text: str) -> str:
    """把手机号 / 学号（8–11 位数字）替换为占位符，其余内容保持可读。"""
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def sanitize_outbound(text: str, limit: int | None = None) -> str:
    """**出域文本的唯一入口**：先脱敏，再按需截断。

    :param text: 原始文本
    :param limit: 截断长度（字符数）；None 表示不截断（评测侧要把完整上下文交给裁判）
    :return: 可直接落库 / 出域的文本
    """
    text = mask_pii(text)
    if limit is not None:
        text = text[:limit]
    return text