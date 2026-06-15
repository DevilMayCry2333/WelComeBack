"""
离线演示 — 不需要LLM端点的简化版本

展示模拟器的核心功能
"""

import numpy as np
import torch
from collections import deque
from universe_state import UniversalState
from physics_constants import PHYSICAL_CONSTANTS, SIMULATION_PARAMS
from latent_predictor import PhysicsPredictor
from metrics import compute_metrics, detect_cosmic_events
from dashboard import CosmicDashboard


def mock_llm_evolution(state: UniversalState) -> tuple:
    """
    模拟LLM生成 (离线版本)

    使用物理规律生成伪嵌入
    """
    # 基于当前状态生成伪嵌入
    base = np.sin(state.time_step * 0.01) * 0.3
    noise = np.random.randn(128) * PHYSICAL_CONSTANTS["hbar"] * 0.5

    # 添加物理约束
    scale_effect = np.log(state.scale_factor + 1) * 0.1
    energy_effect = state.energy_density * 0.2

    embedding = noise + base + scale_effect + energy_effect

    # L2归一化
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding, f"离线模式：宇宙在 t={state.time_step} 演化中"


def run_offline_demo(max_steps: int = 1000):
    """运行离线演示"""
    print("\n" + "="*60)
    print("CFD Physics Simulator - 离线演示模式")
    print("="*60 + "\n")

    # 初始化
    state = UniversalState(
        scale_factor=1e-60,
        energy_density=PHYSICAL_CONSTANTS["rho_crit"],
        entropy=0.0,
        curvature=0.0,
        structure_embedding=np.random.randn(128) * 0.01,
        memory_vector=np.zeros(64),
        consciousness_depth=0.0,
        time_step=0
    )

    predictor = PhysicsPredictor(
        memory_dim=64,
        hidden_dim=128,
        output_dim=128
    )

    # 仪表盘
    dashboard = CosmicDashboard()

    residual_window = deque(maxlen=100)
    state_history = [state.copy()]
    previous_metrics = None

    print(f"开始离线演化... (步数: {max_steps})\n")

    for step in range(1, max_steps + 1):
        # 1. 物理演化
        H = np.sqrt(8 * np.pi / 3 * state.energy_density + 0.7/3)
        state.scale_factor *= (1 + H * 0.01)
        state.energy_density *= (1 - 3 * H * 0.01)
        state.entropy += 0.01 * np.log(state.scale_factor + 1)
        state.time_step = step

        # 2. LLM生成 (离线模拟)
        llm_embedding, text = mock_llm_evolution(state)

        # 融合嵌入
        alpha = 0.3
        state.structure_embedding = alpha * llm_embedding + (1-alpha) * state.structure_embedding
        norm = np.linalg.norm(state.structure_embedding)
        if norm > 0:
            state.structure_embedding /= norm

        # 3. Predictor预测
        with torch.no_grad():
            mem_tensor = torch.FloatTensor(state.memory_vector).unsqueeze(0).unsqueeze(0)
            cons_tensor = torch.FloatTensor([state.consciousness_depth]).unsqueeze(1)
            predicted = predictor(mem_tensor, cons_tensor).numpy().flatten()

        # 4. 残差
        R = state.structure_embedding - predicted
        R_norm = R / (np.linalg.norm(R) + 1e-8)
        residual_window.append(R_norm)

        # 5. 更新记忆和意识
        state.memory_vector += 0.1 * R_norm[:64]
        state.consciousness_depth = np.mean(R_norm**2) * np.std(state.structure_embedding)

        state_history.append(state.copy())
        if len(state_history) > 500:
            state_history = state_history[-250:]

        # 计算指标
        if step % 50 == 0:
            metrics = compute_metrics(residual_window, state_history)
            metrics["time_step"] = step

            dashboard.update(metrics, step)
            dashboard.draw()

            if step % 200 == 0:
                dashboard.print_status(metrics)

            # 检测事件
            events = detect_cosmic_events(metrics, previous_metrics)
            for event in events:
                print(f"\n*** [t={step}] {event} ***\n")
                dashboard.add_event(event, step)

            previous_metrics = metrics

    print("\n离线演示完成!")
    dashboard.save_snapshot('demo_offline_snapshot.png')


if __name__ == '__main__':
    run_offline_demo(1000)
