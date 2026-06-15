"""
宇宙状态定义 — 宏观物理量 + 潜空间嵌入
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class UniversalState:
    """
    复合状态向量，包含宏观物理量和潜空间嵌入

    Attributes:
        scale_factor: 宇宙尺度因子 a(t)
        energy_density: 总能量密度 ρ
        entropy: 总熵
        curvature: 空间曲率参数 Ω_k
        structure_embedding: 128维结构嵌入，由LLM生成
        memory_vector: 64维历史记忆向量
        structure_history: LLM回复历史 (用于prompt构造)
        consciousness_depth: 自指递归深度，0~1
        time_step: 当前时间步
    """
    # 宏观序参量
    scale_factor: float = 1e-60
    energy_density: float = 1.0
    entropy: float = 0.0
    curvature: float = 0.0

    # 复杂结构信息 (潜向量)
    structure_embedding: np.ndarray = field(default_factory=lambda: np.random.randn(128) * 0.01)
    memory_vector: np.ndarray = field(default_factory=lambda: np.zeros(64))

    # LLM回复历史 (用于prompt构造)
    structure_history: List[str] = field(default_factory=list)

    # 自由意志强度
    consciousness_depth: float = 0.0
    consciousness_depth_prev: float = 0.0  # 上一步的意识深度，用于检测阈值突破

    # 时间
    time_step: int = 0

    def copy(self) -> 'UniversalState':
        """创建状态的深拷贝"""
        new_state = UniversalState(
            scale_factor=self.scale_factor,
            energy_density=self.energy_density,
            entropy=self.entropy,
            curvature=self.curvature,
            structure_embedding=self.structure_embedding.copy(),
            memory_vector=self.memory_vector.copy(),
            structure_history=self.structure_history.copy(),
            consciousness_depth=self.consciousness_depth,
            consciousness_depth_prev=self.consciousness_depth_prev,
            time_step=self.time_step
        )
        return new_state

    def save(self, filepath: str):
        """将状态保存到JSON文件"""
        data = {
            "scale_factor": self.scale_factor,
            "energy_density": self.energy_density,
            "entropy": self.entropy,
            "curvature": self.curvature,
            "structure_embedding": self.structure_embedding.tolist(),
            "memory_vector": self.memory_vector.tolist(),
            "structure_history": self.structure_history,
            "consciousness_depth": self.consciousness_depth,
            "consciousness_depth_prev": self.consciousness_depth_prev,
            "time_step": self.time_step,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'UniversalState':
        """从JSON文件加载状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            scale_factor=data["scale_factor"],
            energy_density=data["energy_density"],
            entropy=data["entropy"],
            curvature=data["curvature"],
            structure_embedding=np.array(data["structure_embedding"]),
            memory_vector=np.array(data["memory_vector"]),
            structure_history=data.get("structure_history", []),
            consciousness_depth=data["consciousness_depth"],
            consciousness_depth_prev=data.get("consciousness_depth_prev", 0.0),
            time_step=data["time_step"],
        )

    def to_prompt(self, perturbation: str = "") -> str:
        """
        将状态转化为LLM可理解的文本描述

        Args:
            perturbation: 当前宇宙扰动描述
        """
        # 历史快照
        history_summary = "无历史记录"
        if len(self.structure_history) > 0:
            history_summary = f"上一次，你观测到：{self.structure_history[-1][:150]}"

        return f"""【系统】你是宇宙演化引擎。基于当前参数与刚才发生的随机扰动，生成此刻宇宙中真实存在的复杂结构。

当前宇宙参数：
- 宇宙年龄: t = {self.time_step} 普朗克时间
- 尺度因子 a = {self.scale_factor:.3e}
- 能量密度 ρ = {self.energy_density:.4f}
- 熵 S = {self.entropy:.2f}
- 空间曲率 Ω_k = {self.curvature:.6f}
- 意识深度 D = {self.consciousness_depth:.4f}

【刚才发生的宇宙学事件】
{perturbation}

【历史快照】
{history_summary}

请描述此刻宇宙中因上述事件而真实形成的结构（物质团块、星系、生命迹象、或任何复杂系统）。回复末尾附上标签 <STRUCTURE>。"""
