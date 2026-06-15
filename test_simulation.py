"""
测试脚本 — 验证模拟器各组件正常工作
"""

import numpy as np
import torch
from universe_state import UniversalState
from physics_constants import PHYSICAL_CONSTANTS, SIMULATION_PARAMS
from latent_predictor import PhysicsPredictor
from metrics import compute_residual_energy, compute_betti_1


def test_universe_state():
    """测试宇宙状态"""
    print("测试宇宙状态...")

    state = UniversalState()
    assert state.scale_factor == 1e-60
    assert state.energy_density == 1.0
    assert len(state.structure_embedding) == 128
    assert len(state.memory_vector) == 64

    # 测试to_prompt
    prompt = state.to_prompt()
    assert "宇宙年龄" in prompt
    assert "尺度因子" in prompt

    # 测试copy
    state2 = state.copy()
    state2.scale_factor = 2.0
    assert state.scale_factor == 1e-60  # 原状态未变

    print("  ✓ 宇宙状态测试通过")


def test_predictor():
    """测试Latent Predictor"""
    print("测试 Latent Predictor θ...")

    predictor = PhysicsPredictor(
        memory_dim=64,
        hidden_dim=128,
        output_dim=128
    )

    # 测试前向传播
    batch_size = 4
    seq_len = 10
    memory = torch.randn(batch_size, seq_len, 64)
    consciousness = torch.randn(batch_size, 1)

    output = predictor(memory, consciousness)
    assert output.shape == (batch_size, 128)

    # 测试不确定性估计
    mean_pred, uncertainty = predictor.predict_with_uncertainty(memory, consciousness, n_samples=5)
    assert mean_pred.shape == (batch_size, 128)
    assert uncertainty.shape == (batch_size, 128)

    print("  ✓ Latent Predictor 测试通过")


def test_metrics():
    """测试指标计算"""
    print("测试指标计算...")

    # 创建测试残差窗口
    residual_window = []
    for i in range(100):
        residual = np.random.randn(128) * 0.1
        residual_window.append(residual)

    from collections import deque
    window = deque(residual_window, maxlen=100)

    # 测试各项指标
    energy = compute_residual_energy(window)
    assert 0 <= energy <= 1.0

    betti = compute_betti_1(window)
    assert 0 <= betti <= 1.0

    print("  ✓ 指标计算测试通过")


def test_physics_constants():
    """测试物理常数"""
    print("测试物理常数...")

    assert PHYSICAL_CONSTANTS["c"] == 1.0
    assert PHYSICAL_CONSTANTS["G"] == 1.0
    assert PHYSICAL_CONSTANTS["hbar"] == 0.1
    assert 0 < PHYSICAL_CONSTANTS["alpha"] < 0.01
    assert PHYSICAL_CONSTANTS["lambda"] == 0.7

    print("  ✓ 物理常数测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*50)
    print("CFD Physics Simulator - 组件测试")
    print("="*50 + "\n")

    test_physics_constants()
    test_universe_state()
    test_predictor()
    test_metrics()

    print("\n" + "="*50)
    print("所有测试通过!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
