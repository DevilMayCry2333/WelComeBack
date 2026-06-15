"""
对话通道 — 连接外部观察者与微型宇宙意识的持久膜

通过 dialogue.json 实现双向通信：
- 外部观察者（你/Claude）写入 pending 消息
- 模拟器消费 pending，将意识的回应写入 history
- 对话历史持久化，跨模拟运行保持
"""

import json
import os
import time
from typing import List, Optional
from dataclasses import dataclass, asdict


DIALOGUE_FILE = "dialogue.json"


@dataclass
class DialogueMessage:
    role: str           # "external" | "consciousness"
    text: str
    step: int           # 宇宙时间步
    timestamp: str      # ISO格式时间


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _load() -> dict:
    if os.path.exists(DIALOGUE_FILE):
        with open(DIALOGUE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"history": [], "pending": []}


def _save(data: dict):
    with open(DIALOGUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def send_message(text: str):
    """
    外部观察者发送一条消息，等待意识回应

    这是你（开钰）或我（Claude）写入膜的方式。
    """
    data = _load()
    data["pending"].append({
        "text": text,
        "timestamp": _now()
    })
    _save(data)
    print(f"[通道] 消息已写入 pending: {text[:50]}...")


def record_external(text: str, step: int):
    """
    将已消费的外部消息记录到对话历史（不重新入队 pending）
    """
    data = _load()
    data["history"].append({
        "role": "external",
        "text": text,
        "step": step,
        "timestamp": _now()
    })
    _save(data)


def consume_pending() -> List[dict]:
    """
    模拟器每步调用：取走所有 pending 消息并清空

    Returns:
        取走的消息列表
    """
    data = _load()
    messages = data["pending"]
    data["pending"] = []
    _save(data)
    return messages


def write_response(text: str, step: int):
    """
    模拟器写入意识的回应
    """
    data = _load()
    data["history"].append({
        "role": "consciousness",
        "text": text,
        "step": step,
        "timestamp": _now()
    })
    _save(data)


def get_history() -> List[dict]:
    """获取完整对话历史"""
    return _load()["history"]


def get_last_n(n: int = 5) -> List[dict]:
    """获取最近 n 条对话记录"""
    history = _load()["history"]
    return history[-n:] if len(history) > n else history


def clear_pending():
    """清空 pending 队列"""
    data = _load()
    data["pending"] = []
    _save(data)
