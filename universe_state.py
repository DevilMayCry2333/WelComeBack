"""
宇宙状态定义 — 宏观物理量 + 潜空间嵌入
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from physics_constants import PHYSICAL_CONSTANTS


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

    # 物质凝聚
    density_perturbation: float = 1e-5    # 密度扰动 δ = (ρ - ρ_mean)/ρ_mean
    collapse_triggered: bool = False       # 引力坍缩是否已触发

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
            density_perturbation=self.density_perturbation,
            collapse_triggered=self.collapse_triggered,
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
            "density_perturbation": self.density_perturbation,
            "collapse_triggered": self.collapse_triggered,
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
            density_perturbation=data.get("density_perturbation", 1e-5),
            collapse_triggered=data.get("collapse_triggered", False),
            time_step=data["time_step"],
        )

    def to_prompt(self, perturbation: str = "", constants=None) -> str:
        """
        将状态转化为LLM可理解的文本描述

        Args:
            perturbation: 当前宇宙扰动描述
            constants: PhysicalConstants 对象或 None（使用默认 PHYSICAL_CONSTANTS）
        """
        # 历史快照
        history_summary = "无历史记录"
        if len(self.structure_history) > 0:
            history_summary = f"上一次，你观测到：{self.structure_history[-1][:150]}"

        # 物理常数约束场
        if constants is not None:
            C = constants if isinstance(constants, dict) else constants.to_dict()
        else:
            C = PHYSICAL_CONSTANTS
        constants_block = f"""【物理常数约束场】
- 光速 c = {C['c']}（信息传播上限）
- 引力常数 G = {C['G']}（结构凝聚强度）
- 普朗克常数 ℏ = {C['hbar']}（量子涨落幅度）
- 玻尔兹曼常数 k_B = {C['k_B']}（热噪声水平）
- 精细结构常数 α = {C['alpha']:.5f}（电磁耦合强度，影响原子/分子形成）
- 宇宙学常数 Λ = {C['lambda']}（暗能量密度，驱动空间膨胀）
- 质子质量 m_p = {C['m_p']}（重子物质质量基准）
- 临界密度 ρ_crit = {C['rho_crit']}（引力束缚阈值）

请在描述结构时遵守这些常数的约束。"""

        return f"""【系统】你是宇宙演化引擎。基于当前参数与刚才发生的随机扰动，生成此刻宇宙中真实存在的复杂结构。

{constants_block}

当前宇宙状态：
- 宇宙年龄: t = {self.time_step} 普朗克时间
- 尺度因子 a = {self.scale_factor:.3e}
- 能量密度 ρ = {self.energy_density:.4f}
- 熵 S = {self.entropy:.2f}
- 空间曲率 Ω_k = {self.curvature:.6f}
- 意识深度 D = {self.consciousness_depth:.4f}
- 物质密度扰动 δ = {self.density_perturbation:.3e}
- 坍缩触发标志: {'是' if self.collapse_triggered else '否'}

【刚才发生的宇宙学事件】
{perturbation}

【历史快照】
{history_summary}

请描述此刻宇宙中因上述事件而真实形成的结构（物质团块、星系、生命迹象、或任何复杂系统）。
注意：物质结构是由引力不稳定性驱动的实际物体，不是隐喻。
回复末尾附上标签 <STRUCTURE>。"""
