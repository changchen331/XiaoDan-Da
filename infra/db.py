"""数据库连接基础设施：统一 psycopg3，并由本层保证"用完就关"。

**为什么需要这一层**（两个真实缺陷的收敛点）：

1. **驱动混用**：Checkpointer 必须用 psycopg3（``langgraph-checkpoint-postgres``
   的依赖），而此前长期记忆与高危上报各自用 psycopg2 直连——
   同一个 DSN 上并存两套实现，异常类型（``psycopg2.Error`` / ``psycopg.Error``）
   与事务语义都要分别维护，改一处容易漏另一处。
2. **连接不关闭**：``with psycopg2.connect(...) as conn`` 这样的写法**只提交事务、
   不关闭连接**（psycopg2 / psycopg3 的连接上下文管理器都只负责事务）。
   于是每个请求都会留下一个待 GC 回收的连接，长时间运行会持续泄漏。

因此统一到 psycopg3，并提供 :func:`connection`——退出时显式 ``close()``。

**连接超时为什么写常量而不是配置项**：它是"数据库不可达时不要拖住主流程"的
健康边界，随环境调整没有意义；而每多一个配置项就多一处要在
``.env.example`` / ``docker-compose.yml`` 之间同步的地方（本项目已经因为
配置漂移吃过亏）。将来若真有跨机房远距离数据库需求，再上提为配置。
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from config.settings import settings

#: 建立连接的超时（秒）。宁可快速失败走降级（无记忆 / 上报留待人工补录），
#: 也不要让一次请求卡在网络上：数据库不可达时主回答流程必须照常返回。
#:
#: 取 2 秒而非更宽松的值，是因为**实测超时会被翻倍**：psycopg3 沿 libpq 的
#: 多地址尝试语义，对 ``localhost`` 解析出的 ``::1`` 与 ``127.0.0.1`` 各试一次，
#: 配 5 秒时实测单次调用耗时 10.1s（数据库不可达场景）。本项目数据库是本机 /
#: 容器内服务，2 秒足够；真不可达时最坏 2×2=4 秒。
#: 若将来换成跨机房数据库，这里要连同"多地址翻倍"一起重新评估。
CONNECT_TIMEOUT_SECONDS = 2


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """获取一个业务数据库连接，退出时**显式关闭**。

    事务语义仍由 ``with conn:`` 负责（正常退出提交、异常回滚），
    而 ``finally`` 里的 ``close()`` 保证连接立刻归还，不依赖 GC。

    :return: 可用的 psycopg3 连接（在其上 ``cursor()`` 执行语句）
    """
    conn = psycopg.connect(
        settings.postgres_dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS
    )
    try:
        with conn:  # 事务：正常退出提交，异常回滚
            yield conn
    finally:
        conn.close()