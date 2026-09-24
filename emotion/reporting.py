"""高危上报机制：脱敏 → 写库 → 邮件通知。

隐私保护原则（对应架构文档的数据最小化要求）：
- 上报记录使用内部 user_id，不含真实姓名 / 学号 / 联系方式
- 触发内容与上下文都先**脱敏**（手机号 / 学号 → 占位符）再按需截断：
  触发摘要取前 100 字，上下文仅携带最近 3 轮
- 脱敏口径不在这里定义，统一走 `infra/privacy.sanitize_outbound()`
  （与离线评测送外部裁判前的脱敏共用同一判据，避免"哪类数据算已脱敏"有两种答案）
- 上报数据写入独立的 emotion_alerts 表，与问答数据、记忆数据物理隔离，
  仅授权的心理咨询中心工作人员可查询

可靠性设计：写库失败不阻断关怀回复流程（高危场景下用户优先得到回应，
上报落库失败时打印错误日志，由人工核对补录）。
"""

import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

import psycopg

from config.settings import settings
from infra.db import connection
from infra.privacy import sanitize_outbound

#: 触发摘要的截断长度（字符）。与脱敏同源：它是"出域数据上限"的一部分，
#: 因此不放进 settings（换掉它不会让部署更灵活，只会改变值班老师能看到的判研依据）
TRIGGER_TEXT_LIMIT = 100

#: 上下文只保留最近 3 轮：够人工研判即可
CONTEXT_MAX_ROUNDS = 3

# 高危上报表 DDL：独立于 user_memory 与 Checkpointer 内部表
ALERT_TABLE_DDL = """
                  CREATE TABLE IF NOT EXISTS emotion_alerts
                  (
                      id
                      SERIAL
                      PRIMARY
                      KEY,
                      user_id
                      TEXT
                      NOT
                      NULL, -- 内部 ID（非真实姓名/学号）
                      detected_at
                      TIMESTAMPTZ
                      NOT
                      NULL, -- 触发时间
                      emotion_level
                      TEXT
                      NOT
                      NULL, -- 触发时的情绪级别
                      confidence
                      REAL
                      NOT
                      NULL, -- 检测置信度
                      trigger_summary
                      TEXT, -- 触发内容摘要（前 100 字）
                      recent_context
                      JSONB, -- 最近 3 轮对话上下文
                      referent
                      TEXT -- 危险表达的指向（本人/他人/引用/否定/无）：
                           -- 区分"本人危机"与"代他人求助"，避免把
                           -- "我室友说活不下去"记成提问者本人的高危事件
                  ) \
                  """

# 增量列迁移：CREATE TABLE IF NOT EXISTS 不会给**已存在**的表补列，
# 而升级部署时表是旧的（新列会缺失，INSERT 直接报错）。
# ADD COLUMN IF NOT EXISTS 幂等，每次写库顺手执行即可。
ALERT_REFERENT_MIGRATION = """
                            ALTER TABLE emotion_alerts
                                ADD COLUMN IF NOT EXISTS referent TEXT \
                            """


def _minimize_context(context: list) -> list:
    """上下文出域前的脱敏：只对 `content` 脱敏，保留 role 等结构字段。

    元素形状不保证（`conversation_history` 可由 API 调用方直接传入），故对非字典
    元素也做兜底——上报是安全路径，不该因为一个畸形元素让整轮上报失败。
    """
    minimized: list = []
    for line in context[-CONTEXT_MAX_ROUNDS:]:
        if isinstance(line, dict):
            minimized.append(
                {**line, "content": sanitize_outbound(str(line.get("content", "")))}
            )
        else:
            minimized.append(sanitize_outbound(str(line)))
    return minimized


def build_report_record(
    user_id: str,
    trigger_text: str,
    emotion_level: str,
    emotion_confidence: float,
    recent_context: list,
    referent: str | None = None,
) -> dict:
    """生成脱敏后的上报记录。

    :param user_id: 用户内部 ID
    :param trigger_text: 触发高危判定的原始输入
    :param emotion_level: 情绪级别（"高危"）
    :param emotion_confidence: 检测置信度
    :param recent_context: 最近对话历史（仅取最近 3 轮，且逐条脱敏）
    :param referent: 危险表达的指向；规则命中的高危路径上由上报环节在
        **后台**补齐（见 agent/nodes/report.py），故允许为空
    :return: 可直接入库 / 发邮件的记录字典
    """
    return {
        "user_id": user_id,
        "detected_at": datetime.now().isoformat(),
        "emotion_level": emotion_level,
        "confidence": emotion_confidence,
        # 出域文本一律经 sanitize_outbound：先脱敏、再截断（口径见 infra/privacy.py）
        "trigger_text_summary": sanitize_outbound(
            trigger_text, limit=TRIGGER_TEXT_LIMIT
        ),
        "recent_context": _minimize_context(recent_context),
        "referent": referent,
    }


def submit_report(record: dict) -> int | None:
    """上报执行：写库（立即）+ 邮件通知（已配置时）。

    写库与邮件互不阻断：邮件失败只影响通知时效，不影响记录留痕。

    :return: 入库记录 id；写库失败时为 None（此时不再补指向标注）
    """
    alert_id: int | None = None
    try:
        alert_id = _insert_alert(record)
    except psycopg.Error as db_error:
        # 上报落库失败不抛出：不能因数据库故障中断对用户的关怀回复
        print(f"[reporting] 上报写库失败（需人工核对补录）: {db_error}")

    if settings.ALERT_EMAIL_ENABLED:
        try:
            _send_alert_email(record)
        except (smtplib.SMTPException, OSError) as mail_error:
            print(f"[reporting] 告警邮件发送失败: {mail_error}")

    return alert_id


def annotate_alert_referent(alert_id: int, referent: str) -> None:
    """给已入库的上报记录补"危险表达指向"标注（失败只打日志）。

    单独成一个写操作，是为了让调用方（上报节点）能把它放到后台：
    记录已经落库、邮件已经发出，用户不该为一次标注再等一个超时周期。
    """
    try:
        with connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE emotion_alerts SET referent = %s WHERE id = %s",
                    (referent, alert_id),
                )
    except psycopg.Error as db_error:
        print(f"[reporting] 指向标注写库失败（记录维持未标注）: {db_error}")


def _insert_alert(record: dict) -> int:
    """将上报记录写入独立的 emotion_alerts 表（幂等建表 + 幂等补列）。

    :return: 新记录 id
    """
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(ALERT_TABLE_DDL)
            cursor.execute(ALERT_REFERENT_MIGRATION)
            cursor.execute(
                """
                INSERT INTO emotion_alerts
                (user_id, detected_at, emotion_level, confidence,
                 trigger_summary, recent_context, referent)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    record["user_id"],
                    record["detected_at"],
                    record["emotion_level"],
                    record["confidence"],
                    record["trigger_text_summary"],
                    json.dumps(record["recent_context"], ensure_ascii=False),
                    record.get("referent"),
                ),
            )
            row = cursor.fetchone()
    alert_id = int(row[0]) if row else 0
    print(
        f"[reporting] 高危记录已入库: 用户 {record['user_id']} "
        f"置信度 {record['confidence']:.2f}"
    )
    return alert_id


def _send_alert_email(record: dict) -> None:
    """通过 SMTP SSL 发送告警邮件至心理咨询中心值班邮箱。

    邮件正文只含脱敏摘要，不含完整对话，与库中记录粒度一致。
    """
    body = (
        "【小旦答 - 高危情绪告警】\n\n"
        f"触发时间: {record['detected_at']}\n"
        f"用户内部ID: {record['user_id']}\n"
        f"情绪级别: {record['emotion_level']}（置信度 {record['confidence']:.2f}）\n"
        f"触发内容摘要: {record['trigger_text_summary']}\n\n"
        "请值班老师尽快登录系统查看详情并按流程跟进。"
    )

    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = f"【高危告警】学生情绪风险 - {record['detected_at']}"
    message["From"] = settings.ALERT_SENDER
    message["To"] = settings.ALERT_RECEIVER

    # 465 端口为 SMTP over SSL（加密连接，符合敏感信息传输要求）
    with smtplib.SMTP_SSL(settings.ALERT_SMTP_HOST, settings.ALERT_SMTP_PORT) as server:
        server.login(settings.ALERT_SENDER, settings.ALERT_SMTP_PASSWORD)
        server.sendmail(
            settings.ALERT_SENDER, [settings.ALERT_RECEIVER], message.as_string()
        )

    print(f"[reporting] 告警邮件已发送至 {settings.ALERT_RECEIVER}")
