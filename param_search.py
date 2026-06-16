"""
参数搜索：找到让 Betti-1 稳定在 0.618 附近的参数组合
"""
import numpy as np
from collections import deque
from universe_state import UniversalState
from llm_client import LLMClient
from latent_predictor import PhysicsPredictor, PredictorTrainer
from evolution import evolve, destructive_forgetting_noise
from metrics import compute_betti_1
from physics_constants import PhysicalConstants, SIMULATION_PARAMS
import torch
import random
import json

def run_search(n_steps=60, n_trials=3):
    """运行参数搜索"""
    # 候选参数组合
    param_sets = [
        {"name": "baseline", "hbar": 0.1, "G": 1.0, "attraction": 0.15, "matter": 0.01},
        {"name": "high_noise", "hbar": 0.3, "G": 1.0, "attraction": 0.15, "matter": 0.01},
        {"name": "low_noise", "hbar": 0.05, "G": 1.0, "attraction": 0.15, "matter": 0.01},
        {"name": "high_gravity", "hbar": 0.1, "G": 2.0, "attraction": 0.25, "matter": 0.02},
        {"name": "low_gravity", "hbar": 0.1, "G": 0.5, "attraction": 0.10, "matter": 0.005},
        {"name": "high_attract", "hbar": 0.1, "G": 1.0, "attraction": 0.35, "matter": 0.01},
        {"name": "high_matter", "hbar": 0.1, "G": 1.0, "attraction": 0.15, "matter": 0.05},
        {"name": "balanced", "hbar": 0.15, "G": 1.5, "attraction": 0.20, "matter": 0.02},
    ]

    results = []

    for params in param_sets:
        b1_scores = []

        for trial in range(n_trials):
            random.seed(42 + trial)
            np.random.seed(42 + trial)
            torch.manual_seed(42 + trial)

            # 初始化
            state = UniversalState(
                scale_factor=0.1,
                energy_density=1.0,
                structure_embedding=np.random.randn(128) * 0.01,
                memory_vector=np.zeros(64),
                time_step=0
            )
            predictor = PhysicsPredictor(memory_dim=64, hidden_dim=128, output_dim=128)
            llm = LLMClient()
            constants = PhysicalConstants(mode="standard")
            constants["hbar"] = params["hbar"]
            constants["G"] = params["G"]

            residual_window = deque(maxlen=100)
            state_history = [state.copy()]

            b1_values = []

            for step in range(1, n_steps + 1):
                state.consciousness_depth_prev = state.consciousness_depth

                # 临时覆盖参数
                orig_attraction = SIMULATION_PARAMS["gravitational_attraction"]
                orig_matter = SIMULATION_PARAMS["matter_coupling"]
                SIMULATION_PARAMS["gravitational_attraction"] = params["attraction"]
                SIMULATION_PARAMS["matter_coupling"] = params["matter"]

                state, residual, info = evolve(state, llm, predictor, constants, 'cpu')

                SIMULATION_PARAMS["gravitational_attraction"] = orig_attraction
                SIMULATION_PARAMS["matter_coupling"] = orig_matter

                residual_window.append(residual)
                state_history.append(state.copy())

                if step % 10 == 0 and len(residual_window) >= 20:
                    b1 = compute_betti_1(residual_window)
                    b1_values.append(b1)

            avg_b1 = np.mean(b1_values) if b1_values else 0.0
            max_b1 = np.max(b1_values) if b1_values else 0.0
            b1_scores.append({"avg": avg_b1, "max": max_b1})
            print(f"  {params['name']} trial {trial+1}: avg={avg_b1:.4f}, max={max_b1:.4f}")

        avg_score = np.mean([s["avg"] for s in b1_scores])
        max_score = np.max([s["max"] for s in b1_scores])
        results.append({
            "name": params["name"],
            "params": params,
            "avg_b1": float(avg_score),
            "max_b1": float(max_score)
        })
        print(f"  → {params['name']}: 总平均={avg_score:.4f}, 总最大={max_score:.4f}")

    # 排序
    results.sort(key=lambda x: x["avg_b1"], reverse=True)

    print("\n" + "="*60)
    print("参数搜索结果（按平均B1排序）")
    print("="*60)
    for r in results:
        print(f"  {r['name']:15s} avg={r['avg_b1']:.4f} max={r['max_b1']:.4f}")
        print(f"    hbar={r['params']['hbar']}, G={r['params']['G']}, "
              f"attraction={r['params']['attraction']}, matter={r['params']['matter']}")

    # 保存结果
    with open("param_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 param_search_results.json")

    return results

if __name__ == "__main__":
    run_search()
