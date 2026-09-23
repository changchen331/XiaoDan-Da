"""基础设施层：数据库连接"获取与释放"的契约（不依赖真实数据库）。

为什么这组用例值得单独存在：连接泄漏**平时看不出来**——
``with psycopg.connect(...)`` 只提交事务、不关闭连接，
本地跑一两次毫无异常，只有在长时间运行、连接数耗尽时才暴露。
因此把"退出即关闭"这条语义钉在单测里，而不是依赖人工记得。
"""

import psycopg
import pytest

from infra import db


class _FakeConnection:
    """最小可用的假连接：记录是否被关闭，并支持 with 语句。"""

    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False  # 不吞异常：与真实连接的事务语义一致（异常照样抛给调用方）

    def close(self) -> None:
        self.closed = True


def test_connection_closes_on_normal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常退出时必须显式关闭连接，并带上连接超时。"""
    fake = _FakeConnection()
    captured: dict = {}

    def _fake_connect(dsn: str, **kwargs: object) -> _FakeConnection:
        captured["dsn"] = dsn
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(psycopg, "connect", _fake_connect)

    with db.connection() as conn:
        assert conn is fake
        assert fake.closed is False  # 业务体内连接仍可用

    assert fake.closed is True
    # 连接超时必须存在：数据库"黑洞式不可达"时不能把请求挂死，
    # 必须快速失败进入降级（无记忆 / 上报留待人工补录）
    assert captured["kwargs"]["connect_timeout"] == db.CONNECT_TIMEOUT_SECONDS


def test_connection_closes_and_propagates_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务体抛异常时：连接照样关闭，异常不被吞掉。"""
    fake = _FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: fake)

    with pytest.raises(RuntimeError):
        with db.connection():
            raise RuntimeError("模拟 SQL 执行失败")

    assert fake.closed is True