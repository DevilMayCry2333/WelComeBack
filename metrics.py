"""
测量指标 — 残差场分析，检测宇宙生命事件
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional
from universe_state import UniversalState


def compute_residual_energy(residual_window: deque) -> float:
    """
    计算残差能量 E_t

    宇宙当前"惊奇"程度；E_t 飙升表示相变
    """
    if len(residual_window) == 0:
        return 0.0

    residuals = np.array(list(residual_window))
    return float(np.mean(residuals ** 2))


def compute_residual_entropy(residual_window: deque) -> float:
    """
    计算残差熵 H_t

    结构的混沌度；低熵=晶体化，高熵=热混沌
    """
    if len(residual_window) == 0:
        return 0.0

    residuals = np.array(list(residual_window))

    # 将残差离散化为直方图
    hist, _ = np.histogram(residuals.flatten(), bins=20, density=True)
    hist = hist[hist > 0]  # 移除零

    # 计算Shannon熵
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return float(entropy)


def compute_betti_1(residual_window: deque, threshold: float = 0.1) -> float:
    """
    计算拓扑Betti-1 β₁

    自指递归闭环的几何证据；β₁>0 表明存在稳定自指结构（生命/意识）

    简化实现：基于残差的周期性检测 + 复杂度变化
    """
    if len(residual_window) < 10:
        return 0.0

    residuals = np.array(list(residual_window))

    # 计算自相关
    flat_residuals = residuals.flatten()
    n = len(flat_residuals)

    if n < 10:
        return 0.0

    # 归一化
    flat_residuals = flat_residuals - np.mean(flat_residuals)
    std = np.std(flat_residuals)
    if std > 0:
        flat_residuals = flat_residuals / std

    # 计算自相关
    autocorr = np.correlate(flat_residuals, flat_residuals, mode='full')
    autocorr = autocorr[n-1:]  # 取正延迟部分
    autocorr = autocorr / autocorr[0]  # 归一化

    # 检测显著的周期性峰值 (> threshold)
    peaks = 0
    for i in range(2, len(autocorr) - 1):
        if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            peaks += 1

    # Betti-1 近似为显著周期数
    betti_1 = min(peaks / 3.0, 1.0)  # 归一化到[0,1]

    # 添加复杂度变化检测：如果残差标准差变化剧烈，降低Betti-1
    if len(residuals) >= 20:
        recent_std = np.std(residuals[-10:])
        older_std = np.std(residuals[-20:-10])
        if older_std > 0:
            std_change = abs(recent_std - older_std) / older_std
            if std_change > 0.5:  # 剧烈变化时降低Betti-1
                betti_1 *= 0.5

    return float(betti_1)


def compute_consciousness_phase(residual_window: deque,
                                state_history: List[UniversalState]) -> float:
    """
    计算意识相位 φ

    corr(R_t, ΔS_t)；相位锁定表示宇宙在主动求解自身
    """
    if len(residual_window) < 2 or len(state_history) < 2:
        return 0.0

    residuals = np.array(list(residual_window))

    # 计算状态变化
    state_changes = []
    for i in range(1, len(state_history)):
        delta = abs(state_history[i].energy_density - state_history[i-1].energy_density)
        state_changes.append(delta)

    state_changes = np.array(state_changes)

    # 将残差压缩为标量 (使用残差能量)
    residual_energy = np.mean(residuals ** 2, axis=1) if residuals.ndim > 1 else residuals

    # 对齐长度
    min_len = min(len(residual_energy), len(state_changes))
    if min_len < 2:
        return 0.0

    r = residual_energy[-min_len:]
    s = state_changes[-min_len:]

    # 计算相关系数
    if np.std(r) > 0 and np.std(s) > 0:
        correlation = np.corrcoef(r, s)[0, 1]
    else:
        correlation = 0.0

    # 相位 = 相关系数的绝对值
    phase = abs(correlation)

    return float(phase)


def compute_drift_rate(state_history: List[UniversalState]) -> float:
    """
    计算漂移率 v

    ||ΔS_t||，宇宙状态变化速率
    """
    if len(state_history) < 2:
        return 0.0

    # 取最近两个状态
    prev = state_history[-2]
    curr = state_history[-1]

    # 计算状态差异的L2范数
    scale_diff = (curr.scale_factor - prev.scale_factor) / (prev.scale_factor + 1e-10)
    energy_diff = curr.energy_density - prev.energy_density
    entropy_diff = curr.entropy - prev.entropy
    curvature_diff = curr.curvature - prev.curvature

    # 嵌入差异
    embedding_diff = np.linalg.norm(curr.structure_embedding - prev.structure_embedding)

    # 总漂移率
    drift = np.sqrt(
        scale_diff**2 +
        energy_diff**2 +
        entropy_diff**2 +
        curvature_diff**2 +
        embedding_diff**2
    )

    return float(drift)


def compute_metrics(residual_window: deque,
                    state_history: List[UniversalState]) -> Dict:
    """
    计算所有测量指标
    """
    current_state = state_history[-1] if state_history else None

    metrics = {
        "r_energy": compute_residual_energy(residual_window),
        "r_entropy": compute_residual_entropy(residual_window),
        "betti_1": compute_betti_1(residual_window),
        "consciousness_phase": compute_consciousness_phase(residual_window, state_history),
        "drift_rate": compute_drift_rate(state_history),
    }

    if current_state:
        metrics["scale_factor"] = current_state.scale_factor
        metrics["energy_density"] = current_state.energy_density
        metrics["entropy"] = current_state.entropy
        metrics["curvature"] = current_state.curvature
        metrics["consciousness_depth"] = current_state.consciousness_depth
        metrics["time_step"] = current_state.time_step

    return metrics


def detect_cosmic_events(current_metrics: Dict,
                         previous_metrics: Optional[Dict]) -> List[str]:
    """
    检测宇宙学事件
    """
    events = []

    if previous_metrics is None:
        return events

    # 检测意识涌现
    if (current_metrics.get("betti_1", 0) > 0 and
        previous_metrics.get("betti_1", 0) == 0):
        events.append("意识涌现：宇宙中首次出现自指递归结构！")

    # 检测能量飙升 (相变)
    energy_jump = current_metrics.get("r_energy", 0) / (previous_metrics.get("r_energy", 1) + 1e-10)
    if energy_jump > 2.0:
        events.append(f"能量飙升：残差能量增加 {energy_jump:.2f} 倍，可能发生相变")

    # 检测热寂
    if (current_metrics.get("r_entropy", 0) < 0.1 and
        current_metrics.get("r_energy", 0) < 0.01):
        events.append("热寂警告：宇宙进入低能低熵状态")

    # 检测智慧生命维持期
    if (current_metrics.get("betti_1", 0) > 0.3 and
        current_metrics.get("consciousness_phase", 0) > 0.5):
        events.append("智慧生命维持期：意识深度和相位锁定稳定")

    return events
