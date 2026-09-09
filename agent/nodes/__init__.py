"""Agent 节点集合：从各子模块统一导出，供 graph.py 编排使用。

同时导出四个条件路由函数：
- route_after_emotion：高危 → 上报分支；正常 → 意图路由
- route_after_intent：三条意图分支（简单问答/FAQ 合流到 retrieve）
- route_after_retrieve：FAQ 命中 → 关怀后缀直返；未命中 → 生成流程
- route_after_quality：合格 → 关怀后缀；不合格 → 回退检索重试
- route_after_care_suffix：闲聊不写记忆直接结束；其余写入记忆
"""
from agent.nodes.care_response import care_response
from agent.nodes.care_suffix import care_suffix, route_after_care_suffix
from agent.nodes.chitchat_response import chitchat_response
from agent.nodes.emotion_detect import emotion_detect
from agent.nodes.generate import generate
from agent.nodes.intent_route import intent_route, route_after_intent
from agent.nodes.memory_write import memory_write
from agent.nodes.plan_and_retrieve import plan_and_retrieve
from agent.nodes.quality_check import quality_check, route_after_quality
from agent.nodes.report import report
from agent.nodes.retrieve import retrieve, route_after_retrieve
from agent.nodes.emotion_detect import route_after_emotion

__all__ = [
    "emotion_detect", "report", "care_response", "intent_route",
    "retrieve", "plan_and_retrieve", "care_suffix", "chitchat_response",
    "generate", "quality_check", "memory_write",
    "route_after_emotion", "route_after_intent", "route_after_retrieve",
    "route_after_quality", "route_after_care_suffix",
]
