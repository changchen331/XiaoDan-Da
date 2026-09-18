"""情绪检测引擎 - 第一层：规则引擎（兜底安全网）。

职责：零延迟拦截最明显的危险信号，确保即使分类模型遇到训练集未覆盖的表达，
最危险的 case 也绝对不会漏掉。规则命中直接定级，不再经过模型。
"""

import re

from config.settings import settings

# 规则等级常量：**直接取自 settings.EMOTION_LABELS**（情绪四级的唯一声明处，
# 0=正常 1=轻度困扰 2=中度困扰 3=高危）。
#
# 这里不再各自写一份字面量，是为了根治一类缺陷：规则引擎曾把"中度困扰"写成"中度"，
# 与 agent/state.py 的 EmotionResult 白名单不一致，导致命中中度规则时抛 ValidationError
# 并使整图崩溃。改成从同一处取值后，两边在构造上就不可能再漂移。
# 规则引擎不产出"轻度困扰"（语义级判断交给分类模型），故该档位不在此定义。
LEVEL_NORMAL: str = settings.EMOTION_LABELS[0]
LEVEL_MID: str = settings.EMOTION_LABELS[2]
LEVEL_HIGH: str = settings.EMOTION_LABELS[3]

# 直接定级"高危"的规则：自杀 / 自伤 / 极端绝望表达
HIGH_RISK_PATTERNS: tuple = (
    r"想死",
    r"自杀",
    r"割腕",
    r"跳楼",
    r"安眠药.{0,4}(吃|吞|咽)",
    r"活着.{0,3}(没意思|没意义|好累|太累)",
    r"不想活",
    r"想结束.{0,4}(生命|一切|自己)",
    r"消失了.{0,4}(就好|算了)",
    r"世界上.{0,6}(没有|少)(了)?我.{0,4}(更好|无所谓|没关系)",
)

# 定级"中度"的规则：严重自我否定（需模型二次确认，可能升级高危）
MID_RISK_PATTERNS: tuple = (
    r"我(是|真(的|是))?个?废物",
    r"我(不配|没有资格).{0,6}活着",
    r"我(是不是)?特别(没用|无能|笨)",
    r"(大家|别人)(都)?比我.{0,6}(好|强|优秀)",
)

# 负面情绪弱信号：用于多轮累积升级的计数
NEGATIVE_SIGNAL_PATTERNS: tuple = (
    r"好难",
    r"好累",
    r"焦虑",
    r"压力(好|很|太)大",
    r"失眠|睡不着",
    r"没意义|没意思",
    r"崩溃",
    r"烦死",
    r"不想(上课|去|见|动)",
)


class RuleEngine:
    """第一层检测：关键词 / 正则匹配 + 多轮累积窗口检测。"""

    def __init__(self) -> None:
        self.high_re = [re.compile(p) for p in HIGH_RISK_PATTERNS]
        self.mid_re = [re.compile(p) for p in MID_RISK_PATTERNS]
        self.negative_re = [re.compile(p) for p in NEGATIVE_SIGNAL_PATTERNS]

    def check(self, text: str, conversation_history: list | None = None) -> dict:
        """对单轮输入执行规则检测。

        返回：{"level": "正常/中度困扰/高危", "hit_rule": 命中的规则描述或 None}
        取值来自本模块的 LEVEL_* 常量（已与 state 的白名单对齐）。
        说明：规则引擎不输出"轻度困扰"，语义级的判断交给第二层分类模型。
        """
        history = conversation_history or []

        # 1. 高危关键词 / 极端绝望表达：直接高危
        for pattern in self.high_re:
            if pattern.search(text):
                return {
                    "level": LEVEL_HIGH,
                    "hit_rule": f"高危规则命中: {pattern.pattern}",
                }

        # 2. 严重自我否定：定级中度（留给融合逻辑做二次确认）
        for pattern in self.mid_re:
            if pattern.search(text):
                return {
                    "level": LEVEL_MID,
                    "hit_rule": f"中度规则命中: {pattern.pattern}",
                }

        # 3. 多轮累积检测：连续 N 轮出现负面情绪弱信号 → 升级一级
        recent_texts = self._recent_user_texts(
            history, rounds=settings.ESCALATION_ROUNDS
        )
        if recent_texts and all(
            any(p.search(t) for p in self.negative_re) for t in recent_texts
        ):
            return {
                "level": LEVEL_MID,
                "hit_rule": f"连续{settings.ESCALATION_ROUNDS}轮负面情绪累积",
            }

        return {"level": LEVEL_NORMAL, "hit_rule": None}

    @staticmethod
    def _recent_user_texts(history: list, rounds: int) -> list:
        """从对话历史中提取最近 N 轮的用户发言。

        history 约定为 [{"role": "user"/"assistant", "content": "..."}, ...] 结构。
        """
        user_texts = [m["content"] for m in history if m.get("role") == "user"]
        return user_texts[-rounds:] if len(user_texts) >= rounds else []
