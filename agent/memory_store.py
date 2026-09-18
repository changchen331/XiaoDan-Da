"""长期记忆存储：PostgreSQL user_memory 表的读写。

职责边界（两层记忆体系）：
- 短期记忆：LangGraph Checkpointer 按 thread_id 自动管理（见 agent/graph.py），
  保存完整的多轮对话状态
- 长期记忆：本模块管理，每轮对话结束后存入一句话摘要，
  跨会话持久化，供生成环节做个性化注入（如记住用户上次问过选课）

表结构独立于 Checkpointer 的内部表与 emotion_alerts 上报表，
三个数据域互不干扰，便于分别做权限控制与备份策略。
"""

import psycopg2

from config.settings import settings

# 长期记忆表 DDL（幂等建表）
MEMORY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS user_memory (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT        NOT NULL,
    session_id TEXT        NOT NULL,
    summary    TEXT        NOT NULL,   -- 本轮对话的一句话摘要
    intent     TEXT,                   -- 本轮意图（简单问答/复杂查询/FAQ/闲聊越界）
    created_at TIMESTAMPTZ DEFAULT now()
)
"""


def insert_memory(user_id: str, session_id: str, summary: str, intent: str) -> None:
    """写入一条长期记忆（每轮对话结束时调用一次）。

    写入失败不抛异常：记忆写入属于增强功能，不能阻断主回答流程。
    """
    try:
        with psycopg2.connect(settings.postgres_dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(MEMORY_TABLE_DDL)
                cursor.execute(
                    "INSERT INTO user_memory (user_id, session_id, summary, intent) "
                    "VALUES (%s, %s, %s, %s)",
                    (user_id, session_id, summary, intent),
                )
    except psycopg2.Error as db_error:
        print(f"[memory_store] 长期记忆写入失败（不影响主流程）: {db_error}")


def fetch_recent_memories(user_id: str, limit: int = 5) -> list:
    """查询用户最近的长期记忆（生成环节做个性化上下文注入）。

    :param user_id: 用户内部 ID
    :param limit: 最多返回条数（最近的优先）
    :return: [{"summary", "intent", "created_at"}, ...]，查询失败时返回空列表
    """
    try:
        with psycopg2.connect(settings.postgres_dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT summary, intent, created_at FROM user_memory "
                    "WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit),
                )
                rows = cursor.fetchall()
        return [
            {"summary": row[0], "intent": row[1], "created_at": row[2].isoformat()}
            for row in rows
        ]
    except psycopg2.Error as db_error:
        # 数据库不可用时静默降级：无个性化上下文，回答流程不受影响
        print(f"[memory_store] 长期记忆读取失败（降级为无个性化）: {db_error}")
        return []
