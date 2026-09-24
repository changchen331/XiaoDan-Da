"""基础设施层：数据库连接与**出域脱敏**的契约（不依赖真实数据库 / 外部服务）。

为什么这组用例值得单独存在：

- 连接泄漏**平时看不出来**——``with psycopg.connect(...)`` 只提交事务、不关闭连接，
  本地跑一两次毫无异常，只有在长时间运行、连接数耗尽时才暴露。
  因此把"退出即关闭"这条语义钉在单测里，而不是依赖人工记得。
- 脱敏口径此前在两个模块各写一套（一处正则替换、一处只截断），
  "哪类数据算已脱敏"没有单一判据；现在共用 `infra/privacy.py`，
  把"手机号不与学号混淆""先脱敏再截断"这两条钉在这里。
"""

import psycopg
import pytest

from infra import db
from infra.privacy import mask_pii, sanitize_outbound


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


# ==================== 出域脱敏（infra/privacy.py）====================


def test_mask_pii_keeps_phone_and_student_id_distinct() -> None:
    """手机号必须被标成「手机号」而不是被学号模式吃掉——两者的处置口径不同。

    两条正则都能匹配 11 位数字，所以模式顺序是有语义的：
    若学号模式排在前面，`13812345678` 会被标成 `[学号]`，
    值班老师看到摘要时就分不清"这是联系方式还是学籍号"。
    """
    masked = mask_pii("联系我 13812345678，学号 20211234567")

    assert "[手机号]" in masked
    assert "[学号]" in masked
    assert "13812345678" not in masked
    assert "20211234567" not in masked


def test_sanitize_outbound_masks_before_truncating() -> None:
    """截断不得把号码切成"半截数字"留在文本里（顺序：先脱敏、再截断）。

    若先截断，`xxx13812345678…` 取前 12 字会得到 `xxx138123456`——
    既不是完整的手机号（匹配不上）、也不是空的（仍是可识别的个人信息）。
    """
    out = sanitize_outbound("xxx13812345678" + "尾" * 200, limit=12)

    assert "[手机号]" in out
    assert len(out) <= 12
    assert not any(char.isdigit() for char in out)