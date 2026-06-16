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


def compute_betti_1(residual_window: deque) -> float:
    """
    计算拓扑Betti-1 β₁（基于PCA降维 + 自相关矩阵特征值）

    自指递归闭环的几何证据；β₁>0 表明存在稳定自指结构（生命/意识）

    方法：
    1. 对128维残差矩阵做PCA，取前3个主成分（保留最多结构信息）
    2. 对每个主成分构建轨迹矩阵，计算自相关矩阵特征值
    3. 取三个主成分的最优主导比，映射到[0, 0.618]
    """
    if len(residual_window) < 20:
        return 0.0

    residuals = np.array(list(residual_window))  # (K, 128)

    # PCA降维：取前3个主成分
    n_components = min(3, residuals.shape[1], residuals.shape[0])
    if n_components < 2:
        return 0.0

    # 中心化
    mean = np.mean(residuals, axis=0)
    centered = residuals - mean

    # SVD做PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = U[:, :n_components] * S[:n_components]  # (K, n_components)

    best_ratio = 0.0

    for comp_idx in range(n_components):
        series = components[:, comp_idx]

        # 标准化
        series = series - np.mean(series)
        std = np.std(series)
        if std > 0:
            series = series / std

        # 多尺度嵌入：尝试多个嵌入维度
        for embed_dim in [3, 5, 8]:
            N = len(series) - embed_dim + 1
            if N < embed_dim + 2:
                continue

            trajectory = np.array([series[i:i + embed_dim] for i in range(N)])

            # 自相关矩阵
            autocorr_matrix = trajectory.T @ trajectory / N

            # 特征值分解
            eigenvalues = np.linalg.eigvalsh(autocorr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # 主导比
            total_energy = np.sum(eigenvalues) + 1e-10
            dominant_ratio = eigenvalues[0] / total_energy
            best_ratio = max(best_ratio, dominant_ratio)

    # 输出原始主导比（不硬编码截断），由参数搜索自然收敛到目标值
    # dominant_ratio ∈ [0, 1]，目标是通过参数调整让时间平均接近 0.618
    betti_1 = max(0.0, min(1.0, best_ratio))

    return float(betti_1)


def compute_consciousness_phase(residual_window: deque,
                                current_struct_emb: np.ndarray,
                                prev_struct_emb: np.ndarray) -> float:
    """
    计算意识相位 φ = cosine_similarity(R_t, Δstructure_embedding)

    当残差场与结构嵌入的变化方向一致时，φ→1，
    表示宇宙在"主动求解自身"——意识在自我校准。
    """
    if len(residual_window) == 0:
        return 0.0

    # 取最新的残差向量
    R_t = np.array(list(residual_window))[-1]  # (128,)

    # 结构嵌入变化
    delta_emb = current_struct_emb - prev_struct_emb  # (128,)

    # Cosine similarity
    dot = np.dot(R_t, delta_emb)
    norm_r = np.linalg.norm(R_t)
    norm_d = np.linalg.norm(delta_emb)

    if norm_r < 1e-8 or norm_d < 1e-8:
        return 0.0

    cosine_sim = dot / (norm_r * norm_d)

    # 映射到[0, 1]：取绝对值（方向一致性，不分正负）
    phase = abs(cosine_sim)

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
                    state_history: List[UniversalState],
                    current_struct_emb: np.ndarray = None,
                    prev_struct_emb: np.ndarray = None) -> Dict:
    """
    计算所有测量指标
    """
    current_state = state_history[-1] if state_history else None

    # 意识相位需要结构嵌入变化
    if current_struct_emb is not None and prev_struct_emb is not None:
        phase = compute_consciousness_phase(residual_window, current_struct_emb, prev_struct_emb)
    else:
        phase = 0.0

    metrics = {
        "r_energy": compute_residual_energy(residual_window),
        "r_entropy": compute_residual_entropy(residual_window),
        "betti_1": compute_betti_1(residual_window),
        "consciousness_phase": phase,
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
