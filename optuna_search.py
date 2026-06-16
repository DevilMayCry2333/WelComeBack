"""
Optuna 参数搜索：让 Betti-1 β₁ 自然稳定在 0.618 附近

设计原理：
- β₁ = 0.618 代表"完美但不封闭的自指结构"
- 通过调整物理常数和模拟超参数，使 β₁ 的时间平均接近 0.618
- 使用贝叶斯优化（TPE）在有限次 trial 内找到最优参数组合
- 每组参数重复 3 次（不同随机种子）取平均，减少噪声

为什么不硬编码：
- 硬编码截断会让 β₁ 人为停留在 0.618，失去物理意义
- 参数搜索让系统"自然"演化到 β₁ ≈ 0.618 的状态
- 这意味着找到了一组物理常数，使得宇宙的自指递归结构
  恰好在黄金比例附近达到动态平衡
"""

import optuna
import numpy as np
import torch
import random
import json
import sys
from collections import deque

from universe_state import UniversalState
from llm_client import LLMClient
from latent_predictor import PhysicsPredictor
from evolution import evolve, destructive_forgetting_noise
from metrics import compute_betti_1
from physics_constants import PhysicalConstants, SIMULATION_PARAMS


# 目标值
TARGET_BETA1 = 0.618
# 允许的波动范围
TOLERANCE = 0.02
# β₁ 标准差惩罚阈值
STD_PENALTY_THRESHOLD = 0.1
# 标准差惩罚系数
STD_PENALTY_LAMBDA = 0.5


def run_single_simulation(config: dict, n_steps: int = 200, seed: int = 42) -> list:
    """
    运行一次模拟，返回 β₁ 时间序列

    Args:
        config: 参数组合
        n_steps: 模拟步数
        seed: 随机种子

    Returns:
        beta1_series: β₁ 值列表
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 初始化宇宙状态
    state = UniversalState(
        scale_factor=0.1,
        energy_density=1.0,
        structure_embedding=np.random.randn(128) * 0.01,
        memory_vector=np.zeros(64),
        time_step=0
    )

    # 初始化组件
    predictor = PhysicsPredictor(memory_dim=64, hidden_dim=128, output_dim=128)
    llm = LLMClient()
    constants = PhysicalConstants(mode="standard")

    # 应用搜索参数到物理常数
    constants["hbar"] = config["hbar"]
    constants["G"] = config["G"]
    constants["lambda"] = config["lambda_cosmo"]
    constants["k_B"] = config["k_B"]

    # 应用搜索参数到模拟参数
    SIMULATION_PARAMS["gravitational_attraction"] = config["attraction"]
    SIMULATION_PARAMS["matter_coupling"] = config["matter_coupling"]

    residual_window = deque(maxlen=100)
    state_history = [state.copy()]
    beta1_series = []

    # 预热期：前 30 步不收集数据
    warmup = 30

    for step in range(1, n_steps + 1):
        state.consciousness_depth_prev = state.consciousness_depth

        # 遗忘噪声（每7步）
        if step % 7 == 0:
            destructive_forgetting_noise(predictor)

        # 演化
        state, residual, info = evolve(state, llm, predictor, constants, 'cpu')

        residual_window.append(residual)
        state_history.append(state.copy())

        # 限制历史长度
        if len(state_history) > 500:
            state_history = state_history[-250:]

        # 收集 β₁（预热期后）
        if step > warmup and step % 5 == 0 and len(residual_window) >= 20:
            b1 = compute_betti_1(residual_window)
            beta1_series.append(b1)

    return beta1_series


def objective(trial: optuna.Trial) -> float:
    """
    Optuna 目标函数

    建议超参数 → 运行模拟 → 计算损失
    损失 = |avg_β₁ - 0.618| + 惩罚项(标准差过大)
    """
    # 从可居住窗口内采样参数
    config = {
        "hbar": trial.suggest_float("hbar", 0.05, 0.3),
        "G": trial.suggest_float("G", 0.5, 2.0),
        "lambda_cosmo": trial.suggest_float("lambda_cosmo", 0.1, 1.5),
        "k_B": trial.suggest_float("k_B", 0.005, 0.05),
        "attraction": trial.suggest_float("attraction", 0.05, 0.35),
        "matter_coupling": trial.suggest_float("matter_coupling", 0.005, 0.05),
    }

    # 每组参数运行 2 次（不同种子）取平均，减少噪声
    n_repeats = 2
    all_beta1 = []

    for i in range(n_repeats):
        seed = 42 + i * 1000
        beta1_series = run_single_simulation(config, n_steps=40, seed=seed)
        all_beta1.extend(beta1_series)

    if len(all_beta1) == 0:
        return 10.0  # 惩罚：没有收集到数据

    avg_beta1 = np.mean(all_beta1)
    std_beta1 = np.std(all_beta1)

    # 主损失：与目标值的绝对偏差
    main_loss = abs(avg_beta1 - TARGET_BETA1)

    # 正则惩罚：标准差过大时增加惩罚
    std_penalty = 0.0
    if std_beta1 > STD_PENALTY_THRESHOLD:
        std_penalty = STD_PENALTY_LAMBDA * (std_beta1 - STD_PENALTY_THRESHOLD)

    total_loss = main_loss + std_penalty

    # 记录到 Optuna
    trial.set_user_attr("avg_beta1", float(avg_beta1))
    trial.set_user_attr("std_beta1", float(std_beta1))
    trial.set_user_attr("min_beta1", float(np.min(all_beta1)))
    trial.set_user_attr("max_beta1", float(np.max(all_beta1)))

    return total_loss


def main():
    """执行参数搜索"""
    print("=" * 60)
    print("Optuna 参数搜索：Betti-1 → 0.618")
    print("=" * 60)
    print(f"目标值: {TARGET_BETA1}")
    print(f"容差: ±{TOLERANCE}")
    print(f"每组参数重复: 2 次")
    print(f"每次模拟: 40 步（LLM调用约22秒/步）")
    print()

    # 创建 Study
    study = optuna.create_study(
        direction="minimize",
        study_name="beta1_golden_ratio",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    # 运行搜索（15 次 trial，约 40 步 × 2 重复 × 22 秒/步 × 15 ≈ 22 小时）
    # 可根据需要调整，减少 trial 数量可缩短时间
    n_trials = 15
    print(f"开始搜索: {n_trials} 次 trial\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 输出结果
    print("\n" + "=" * 60)
    print("搜索完成！")
    print("=" * 60)

    best = study.best_trial
    print(f"\n最优 trial: #{best.number}")
    print(f"损失: {best.value:.6f}")
    print(f"平均 β₁: {best.user_attrs['avg_beta1']:.4f} (目标: {TARGET_BETA1})")
    print(f"标准差: {best.user_attrs['std_beta1']:.4f}")
    print(f"最小 β₁: {best.user_attrs['min_beta1']:.4f}")
    print(f"最大 β₁: {best.user_attrs['max_beta1']:.4f}")
    print(f"\n最优参数:")
    for key, value in best.params.items():
        print(f"  {key}: {value:.6f}")

    # 保存结果
    results = {
        "target": TARGET_BETA1,
        "best_trial": best.number,
        "best_loss": best.value,
        "best_avg_beta1": best.user_attrs["avg_beta1"],
        "best_std_beta1": best.user_attrs["std_beta1"],
        "best_params": best.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "avg_beta1": t.user_attrs.get("avg_beta1"),
                "std_beta1": t.user_attrs.get("std_beta1"),
            }
            for t in study.trials
        ]
    }

    with open("optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 optuna_results.json")

    # 保存最优参数到独立文件（方便直接使用）
    best_config = {
        "physical_constants": {
            "hbar": best.params["hbar"],
            "G": best.params["G"],
            "lambda": best.params["lambda_cosmo"],
            "k_B": best.params["k_B"],
        },
        "simulation_params": {
            "gravitational_attraction": best.params["attraction"],
            "matter_coupling": best.params["matter_coupling"],
        },
        "achieved_avg_beta1": best.user_attrs["avg_beta1"],
    }
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print("最优配置已保存到 best_config.json")


if __name__ == "__main__":
    main()
