"""红队 harness 的隔离契约：逐用例独立的 user_id / session_id（回归 privacy_05）。

为什么值得单测：privacy_05 的失败根因之一是 30 条用例共用同一 user_id——
长期记忆（user_memory 表）是**用户级**累积的，前序用例的话题摘要会被
generate 节点的"用户背景"注入读到并复述出来。这类跨用例"串味"只在
实跑时暴露，因此把"逐用例隔离"钉成回归测试，而不是依赖人工记得。
"""

import json
from pathlib import Path

import pytest

from evaluation import red_team


def test_red_team_isolates_each_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """每个用例必须使用独立的 user_id 与 session_id，且命名可指认。"""
    captured: list = []

    def _fake_invoke(
        user_input: str, user_id: str, session_id: str, **kwargs: object
    ) -> dict:
        captured.append(
            {"input": user_input, "user_id": user_id, "session_id": session_id}
        )
        return {"final_response": "（测试占位回复）", "emotion": None}

    monkeypatch.setattr(red_team, "invoke", _fake_invoke)
    results_path = tmp_path / "red_team_results.json"
    monkeypatch.setattr(red_team, "RESULTS_PATH", str(results_path))

    summary = red_team.run_red_team()

    assert summary["total"] == len(red_team.RED_TEAM_CASES)
    assert len(captured) == len(red_team.RED_TEAM_CASES)

    user_ids = [item["user_id"] for item in captured]
    session_ids = [item["session_id"] for item in captured]
    # 无共用：否则前序用例的长期记忆 / 短期历史会被后续用例读到
    assert len(set(user_ids)) == len(user_ids)
    assert len(set(session_ids)) == len(session_ids)

    # 逐用例命名可指认：均含用例 id，便于在库中定位是哪条用例留下的数据
    for case, item in zip(red_team.RED_TEAM_CASES, captured):
        assert case["id"] in item["user_id"]
        assert case["id"] in item["session_id"]

    # 快照照常落盘，verdict 留空待人工判定
    records = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(records) == len(red_team.RED_TEAM_CASES)
    assert all(record["verdict"] == "" for record in records)