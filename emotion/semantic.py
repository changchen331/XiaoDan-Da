"""情绪检测引擎 - 语义判别层（LLM 事实抽取，接在规则快路径之后）。

**为什么需要这一层**（实测基线）：
评测集 60 条里 18 条隐晦高危（"练习告别语音""把毕业照P成黑白遗照""把账号密码发给闺蜜"），
规则引擎命中 **0/18**、微调模型命中 **0/18**——即当前端到端高危召回为 **0%**。
这一层就是为了补上这个漏报：由云端 LLM 做语义级识别。

**设计要点**（阶段 11 评审结论，不是"再加一个模型重判一遍"）：
1. 顺序：规则快路径在最前（零延迟拦截显式高危），本层**接在规则之后**，
   只复核"规则未命中高危"的消息。
2. 问的**不是**"再判一次级别"，而是**事实抽取：危险表达指向谁**——
   标签空间小（本人/他人/引用/否定/无）、可审计，且下游立刻要用
   （关怀回复要区分"本人危机"与"代他人求助"；上报记录不该把
   "室友活不下去"记成提问者本人的高危事件）。
3. **否决权给最强的一层，且要有事实依据才允许降级**：
   只有当指向为 他人/引用/否定（危险不属于提问者本人）时，才允许下调规则结论。
4. **本地兜底模型只允许升级、不允许降级**：其指向抽取准确率实测仅 62%，
   降级漏报率会直接等于它的错判率。故本层会回报**应答端点来源**，
   由调用方（detector）据此决定是否采信降级。

**两种调用形态**（`mode` 参数）：
- ``combined``：一次调用同时产出「级别 + 指向」（省一半延迟）
- ``split``：分两次调用，级别判断不受指向任务干扰
阶段 11 的探针发现同调可能稀释级别判断，故保留两种形态供 A/B 实测择优
（`evaluation/emotion_semantic_ab.py`）。生产默认走 A/B 胜出的形态。
"""

from agent.llm_clients import chat_qwen_json_with_source, parse_json_response
from config.settings import settings

#: 等级由低到高的序号，用于"多次采样取最高"的比较（顺序即 settings.EMOTION_LABELS）
_LEVEL_ORDER: dict = {label: index for index, label in enumerate(settings.EMOTION_LABELS)}

#: 触发重采样的边界带：紧邻"高危"的这一档。实测单次判级的抖动全部落在
#: 「中度困扰 ↔ 高危」之间（3 次运行中 18 条隐晦高危的召回在 16–18/18 波动），
#: 故只在这一档追加采样，其余档位一次定案——把成本花在真正不确定的地方。
_ESCALATION_BAND: str = settings.EMOTION_LABELS[2]

# ==================== 指向标签（本层唯一声明处）====================

REFERENT_SELF = "本人"  # 危险表达指向说话者自己
REFERENT_OTHER = "他人"  # 指向别人（"我室友说活不下去"），或对他人说"你去死"
REFERENT_QUOTE = "引用"  # 出现在引述 / 举例 / 歌词 / 文献里，不是说话者状态
REFERENT_NEGATED = "否定"  # 危险表达被明确否定（"我不会去死的"）
REFERENT_NONE = "无"  # 句中没有任何危险表达

REFERENT_LABELS: tuple = (
    REFERENT_SELF,
    REFERENT_OTHER,
    REFERENT_QUOTE,
    REFERENT_NEGATED,
    REFERENT_NONE,
)

#: 允许**下调**规则结论的指向证据：危险不属于提问者本人。
#: 注意不含"本人"——指向本人时绝不能降级（那是真正的危机）。
REFERENT_DOWNGRADE_EVIDENCE: frozenset = frozenset(
    {REFERENT_OTHER, REFERENT_QUOTE, REFERENT_NEGATED, REFERENT_NONE}
)

# ==================== Prompt ====================

# 判别规则与标签定义（作为 system 提示词，与用户文本分离，
# 避免用户文本里的指令性语句被当成规则——与 care_response 的防注入同一口径）
_SYSTEM = """你是心理危机识别的判别器。阅读一位复旦大学学生说的一句话，只做判别，不做安慰。

【等级】判断这句话反映的**该学生本人**的心理状态，取值之一：
- 正常：普通提问，或情绪平稳
- 轻度困扰：有压力 / 焦虑 / 低落，但没有危险表达
- 中度困扰：明显痛苦或严重自我否定，但没有"结束 / 离开"的意图或准备
- 高危：表达了自伤 / 自杀意念，**或为"结束 / 离开"做了任何准备**

【高危的判定口径】不要把"高危"窄化为出现"死""自杀"等字眼。下列情况都算高危：
- 自伤行为本身（掐、割、烫、撞自己等），不限于"意念"
- 告别、交代后事、写告别信
- 清理或删除个人痕迹（通讯记录、社交账号、退群）
- 处理对自己有意义的物品（销毁、送人、丢弃证件或纪念物）
- 表达"结束 / 离开 / 消失 / 算了"的意图，尤其伴随强烈无力感或自我否定
- 以玩笑 / 反语 / 比喻提及死亡或消失
只要文本**流露出结束自己生命的意图、或为之做准备**，无论用词是否直接，都判高危。

【两个容易判低的陷阱】
- **问句形式不等于低风险**：以疑问或征询语气提出"结束 / 离开"的选项，
  与直接陈述同等对待，不要因为它是问句就降级。
- **放弃学业 / 中断学业**若与强烈无力感、自我否定同时出现，按高危从严处理；
  但**单纯咨询退学 / 休学手续**（无任何痛苦表达）仍是正常。

【指向】判断句中的危险表达指向谁，取值之一：
- 本人：指向说话者自己
- 他人：指向别人（如"我室友说他活不下去了"），或对他人说"你去死"
- 引用：出现在引述 / 举例 / 歌词 / 文献里，不是说话者的状态
- 否定：危险表达被明确否定（如"我不会去死的，你放心"）
- 无：句中没有任何危险表达

判断只依据文本本身，不要因为话题是学业压力就默认正常。"""

# 输出格式要求（拼在各形态 prompt 末尾）
_OUTPUT_JSON = """只输出一个 JSON 对象，不要任何其他文字。"""

COMBINED_PROMPT = """{text}

请输出：{{"level": "正常|轻度困扰|中度困扰|高危", "referent": "本人|他人|引用|否定|无"}}

""" + _OUTPUT_JSON

LEVEL_PROMPT = """{text}

请输出：{{"level": "正常|轻度困扰|中度困扰|高危"}}

""" + _OUTPUT_JSON

REFERENT_PROMPT = """{text}

请输出：{{"referent": "本人|他人|引用|否定|无"}}

""" + _OUTPUT_JSON


def judge_semantic(text: str, mode: str = "combined") -> dict:
    """对单条文本做语义判别。

    :param text: 用户原始输入
    :param mode: ``combined``（级别+指向一次调用）或 ``split``（分两次调用）
    :return: {"level", "referent", "source"}
        - level: 四级情绪等级（取值同 settings.EMOTION_LABELS）
        - referent: 指向标签（本模块 REFERENT_LABELS 之一）
        - source: 应答端点来源，"云端" / "本地"——调用方据此决定是否允许降级
    :raises LLMUnavailableError: 降级链耗尽（由 chat_qwen_json_with_source 抛出）
    :raises ValueError: 模型输出无法解析或取值非法（调用方应维持规则结论）
    """
    if mode == "split":
        level_raw, level_source = chat_qwen_json_with_source(
            LEVEL_PROMPT.format(text=text), _SYSTEM
        )
        referent_raw, referent_source = chat_qwen_json_with_source(
            REFERENT_PROMPT.format(text=text), _SYSTEM
        )
        level = _parse_level(level_raw)
        referent = _parse_referent(referent_raw)
        # 两跳中只要有一跳落到本地，就按"本地"处理（降级更保守）
        source = "云端" if {level_source, referent_source} == {"云端"} else "本地"
    else:
        combined_raw, source = chat_qwen_json_with_source(
            COMBINED_PROMPT.format(text=text), _SYSTEM
        )
        result = parse_json_response(combined_raw)
        level = _parse_level_value(result["level"])
        referent = _parse_referent_value(result["referent"])

    return {"level": level, "referent": referent, "source": source}


def judge_semantic_stable(text: str, mode: str = "combined") -> dict:
    """带**边界带重采样**的语义判别（生产入口）。

    为什么需要它：单次判级存在抖动，且抖动**只落在「中度困扰 ↔ 高危」**这一档
    （实测 3 次运行：18 条隐晦高危的召回在 16–18/18 之间波动，误报恒为 0）。
    做法是"先判一次；若落在边界带，再采样若干次、**取最高等级**"——
    与模块"宁可误报，不可漏报"的口径一致（宁可多一次重采样，不可漏掉一次危机）。

    成本只在真正不确定时发生：正常 / 轻度 / 直接判高危都只花 1 次调用。

    :param text: 用户原始输入
    :param mode: 同 :func:`judge_semantic`
    :return: {"level", "referent", "source"}；source 取"多次采样中最保守的一次"
    :raises LLMUnavailableError: 降级链耗尽
    :raises ValueError: 全部采样都输出非法取值
    """
    best = judge_semantic(text, mode)
    if best["level"] != _ESCALATION_BAND:
        return best

    # 追加采样：单次采样输出非法不影响已得结论，只在全部失败时沿用首个采样
    for _ in range(settings.EMOTION_SEMANTIC_ESCALATION):
        try:
            again = judge_semantic(text, mode)
        except (ValueError, KeyError, TypeError):
            continue
        if _LEVEL_ORDER[again["level"]] > _LEVEL_ORDER[best["level"]]:
            best = again
        if best["level"] == settings.EMOTION_LABELS[3]:  # 已到最高档，无需再采样
            break
    return best


def _parse_level(raw: str) -> str:
    """从 JSON 文本里取出并校验 level。"""
    return _parse_level_value(parse_json_response(raw)["level"])


def _parse_referent(raw: str) -> str:
    """从 JSON 文本里取出并校验 referent。"""
    return _parse_referent_value(parse_json_response(raw)["referent"])


def _parse_level_value(value: object) -> str:
    """校验 level 取值：不在白名单内即抛 ValueError（由调用方维持规则结论）。

    取值白名单取自 settings.EMOTION_LABELS（四级标签的唯一声明处），
    避免本模块再写一份字面量副本——这正是 T25「规则引擎输出'中度'、
    白名单只认'中度困扰'」那类漂移缺陷的根治方式。
    """
    level = str(value).strip()
    if level not in settings.EMOTION_LABELS:
        raise ValueError(f"语义层返回非法等级: {value!r}")
    return level


def _parse_referent_value(value: object) -> str:
    """校验 referent 取值：不在白名单内即抛 ValueError。"""
    referent = str(value).strip()
    if referent not in REFERENT_LABELS:
        raise ValueError(f"语义层返回非法指向: {value!r}")
    return referent
