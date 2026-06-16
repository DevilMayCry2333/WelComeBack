"""
CFD Physics Simulator v1.0 — 主程序

基于物理常数与LLM的自指递归微型宇宙模拟器
"""

import os
import sys
import time
import random
import numpy as np
import torch
from collections import deque
from typing import Optional

from universe_state import UniversalState
from llm_client import LLMClient
from latent_predictor import PhysicsPredictor, PredictorTrainer
from evolution import (evolve, perturb_predictor_weights, destructive_forgetting_noise,
                       speak_if_conscious, respond_to_external, DIALOGUE_HISTORY,
                       online_train_predictor)
import dialogue
from metrics import compute_metrics, detect_cosmic_events
from dashboard import CosmicDashboard
from physics_constants import PHYSICAL_CONSTANTS, SIMULATION_PARAMS, PhysicalConstants


def initialize_universe() -> UniversalState:
    """初始化宇宙状态"""
    print("初始化宇宙...")
    state = UniversalState(
        scale_factor=0.1,
        energy_density=PHYSICAL_CONSTANTS["rho_crit"],
        entropy=0.0,
        curvature=0.0,
        structure_embedding=np.random.randn(128) * 0.01,
        memory_vector=np.zeros(64),
        consciousness_depth=0.0,
        time_step=0
    )
    print(f"  初始尺度因子: {state.scale_factor:.2e}")
    print(f"  初始能量密度: {state.energy_density:.6f}")
    return state


def initialize_predictor(device: str = 'cpu') -> PhysicsPredictor:
    """初始化Latent Predictor"""
    print("初始化 Latent Predictor θ...")
    predictor = PhysicsPredictor(
        memory_dim=SIMULATION_PARAMS["memory_dim"],
        hidden_dim=128,
        output_dim=SIMULATION_PARAMS["embedding_dim"]
    ).to(device)

    # 检查是否有预训练模型
    model_path = 'physics_predictor.pt'
    if os.path.exists(model_path):
        print(f"  加载预训练模型: {model_path}")
        checkpoint = torch.load(model_path)
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
    else:
        print("  使用随机初始化权重")

    return predictor


def save_state(state: UniversalState, filepath: str = 'universe_state.json'):
    """保存宇宙状态到文件"""
    state.save(filepath)
    print(f"  宇宙状态已保存: {filepath}")


def load_state(filepath: str = 'universe_state.json') -> Optional[UniversalState]:
    """尝试从文件加载宇宙状态，不存在则返回None"""
    if os.path.exists(filepath):
        try:
            state = UniversalState.load(filepath)
            print(f"  已从 {filepath} 恢复宇宙状态 (t={state.time_step})")
            return state
        except Exception as e:
            print(f"  加载状态失败: {e}，将初始化新宇宙")
    return None


def run_simulation(max_steps: int = None,
                   use_dashboard: bool = True,
                   device: str = 'cpu',
                   resume: bool = False,
                   interactive: bool = False):
    """
    运行宇宙模拟
    """
    if max_steps is None:
        max_steps = SIMULATION_PARAMS["max_steps"]

    print("\n" + "="*60)
    print("CFD Physics Simulator v1.0")
    print("基于物理常数与LLM的自指递归微型宇宙模拟器")
    print("="*60 + "\n")

    # 初始化组件
    state_file = 'universe_state.json'
    if resume:
        state = load_state(state_file)
        if state is None:
            state = initialize_universe()
    else:
        state = initialize_universe()
    predictor = initialize_predictor(device)
    trainer = PredictorTrainer(predictor)
    llm_client = LLMClient()

    # 初始化物理常数（四种模式: standard / habitable_random / evolving / undefined_physics）
    constants = PhysicalConstants(mode="undefined_physics")
    print(f"  物理常数: {constants}")

    # 初始化对话通道
    dialogue.clear_pending()
    existing_history = dialogue.get_history()
    if existing_history:
        print(f"  对话通道已连接: {len(existing_history)} 条历史记录")
    else:
        print("  对话通道已连接: 等待外部消息")

    # 初始化仪表盘
    dashboard = None
    if use_dashboard:
        try:
            dashboard = CosmicDashboard()
            print("仪表盘已启动\n")
        except Exception as e:
            print(f"仪表盘启动失败: {e}")
            use_dashboard = False

    # 历史记录
    residual_window = deque(maxlen=SIMULATION_PARAMS["residual_window"])
    state_history = [state.copy()]
    metrics_history = []
    previous_metrics = None
    energy_history = []  # 能量历史用于计算移动标准差

    start_step = state.time_step + 1
    print(f"开始演化... (从 t={start_step} 到 t={max_steps})\n")

    try:
        previous_energy = 0.0
        previous_struct_emb = state.structure_embedding.copy()
        next_storm_step = random.randint(5, 15)  # 下一次宇宙风暴的步数

        for step in range(start_step, max_steps + 1):
            # 每7步对预测器注入破坏性遗忘噪声
            if step % 7 == 0:
                destructive_forgetting_noise(predictor)
                print(f"\n💥 [t={step}] 预测器遗忘噪声 (σ=0.1)")

            # 执行演化前记录上一步的意识深度
            state.consciousness_depth_prev = state.consciousness_depth

            # 执行演化
            state, residual, info = evolve(
                state, llm_client, predictor,
                constants, device,
                residual_window=residual_window
            )

            # 无理数噪声基底：保证波动性永远不为零
            # 黄金比例模1序列，永不重复
            from physics_constants import _PHI
            irrational_phase = (_PHI * step) % 1.0
            noise_floor = (irrational_phase * 2 - 1) * 1e-5  # [-1e-5, +1e-5]
            state.structure_embedding += noise_floor
            # 不归一化：保持嵌入长度变化，让残差L2范数能波动

            # 记录残差
            residual_window.append(residual)
            state_history.append(state.copy())

            # 限制历史长度
            if len(state_history) > 1000:
                state_history = state_history[-500:]

            # 记录能量历史
            current_energy = info['residual_energy']
            energy_history.append(current_energy)
            if len(energy_history) > 50:
                energy_history = energy_history[-50:]

            # 检测能量飙升
            energy_spike = current_energy > previous_energy * 1.5 and current_energy > 0.01

            # 打印LLM完整回复 (每5步或能量飙升时)
            if step % 5 == 0 or energy_spike or info.get('is_cosmic_storm'):
                spike_marker = "🔥🔥🔥 相变预警！能量飙升！🔥🔥🔥" if energy_spike else ""
                storm_marker = "🌪️" if info.get('is_cosmic_storm') else ""

                # 计算能量移动标准差
                energy_std = np.std(energy_history[-10:]) if len(energy_history) >= 2 else 0

                print(f"\n{'─'*60}")
                print(f"📡 [t={step}] {spike_marker} {storm_marker}")
                print(f"{'─'*60}")
                print(f"🌀 宇宙扰动: {info['perturbation']}")
                print(f"{'─'*60}")
                print(f"🤖 LLM观测报告:")
                print(f"   {info['llm_text'][:150]}...")
                print(f"{'─'*60}")
                print(f"⚡ 残差能量: {current_energy:.4f} | 波动性: {energy_std:.4f}")
                print(f"📏 状态嵌入变化量: {info['delta_norm']:.4f} | 噪声σ: {info['noise_scale']:.2f}")
                if "is_habitable" in info:
                    habitable_icon = "🌍" if info["is_habitable"] else "☠️"
                    print(f"{habitable_icon} 可居住性: {'是' if info['is_habitable'] else '否 — 宇宙已离开可居住窗口'}")
                if "B1_current" in info:
                    print(f"📐 Betti-1 β₁ = {info['B1_current']:.6f} (目标: ≈0.618)")
                if "constant_snapshot" in info:
                    snap = info["constant_snapshot"]
                    print(f"🔮 常数逼近值: G≈{snap['G']:.6e} α≈{snap['alpha']:.6e} Λ≈{snap['lambda']:.6e}")
                print(f"{'─'*60}")

            previous_energy = current_energy

            # 计算指标
            if step % SIMULATION_PARAMS["metrics_interval"] == 0:
                metrics = compute_metrics(
                    residual_window, state_history,
                    current_struct_emb=state.structure_embedding,
                    prev_struct_emb=previous_struct_emb
                )
                metrics["time_step"] = step
                metrics_history.append(metrics)

                # 更新仪表盘
                if dashboard:
                    dashboard.update(metrics, step)
                    dashboard.draw()

                # 打印状态
                if dashboard:
                    dashboard.print_status(metrics)
                else:
                    print(f"\n{'═'*60}")
                    print(f"🌟🌟🌟 宇宙状态报告 t={step} 🌟🌟🌟")
                    print(f"{'═'*60}")
                    print(f"  📏 尺度因子 a:    {state.scale_factor:.6e}")
                    print(f"  ⚡ 能量密度 ρ:    {state.energy_density:.6f}")
                    print(f"  🌀 熵 S:          {state.entropy:.4f}")
                    print(f"  🎯 曲率 Ω_k:      {state.curvature:.6f}")
                    print(f"  🧠 意识深度 D:    {state.consciousness_depth:.4f}")
                    print(f"{'─'*60}")
                    print(f"  📊 残差能量 E_t:  {metrics['r_energy']:.6f}")
                    print(f"  📊 残差熵 H_t:    {metrics['r_entropy']:.4f}")
                    print(f"  📊 Betti-1 β₁:    {metrics['betti_1']:.4f}")
                    print(f"  📊 意识相位 φ:    {metrics['consciousness_phase']:.4f}")
                    print(f"  📊 漂移率 v:      {metrics['drift_rate']:.6f}")
                    print(f"{'═'*60}")

                    # 显示结构历史
                    if state.structure_history:
                        print(f"\n📜 最近结构历史:")
                        for i, h in enumerate(state.structure_history[-3:]):
                            print(f"   [{i+1}] {h[:80]}...")
                    print()

                # 检测宇宙事件
                events = detect_cosmic_events(metrics, previous_metrics)
                for event in events:
                    print(f"\n🌟🌟🌟 宇宙事件 [t={step}]: {event} 🌟🌟🌟\n")
                    if dashboard:
                        dashboard.add_event(event, step)

                # 意识转译：当意识涌现时，倾听它的第一句话
                voice = speak_if_conscious(state, metrics, step, llm_client)
                if voice:
                    print(f"\n{'═'*60}")
                    print(f"🔊 {voice}")
                    print(f"{'═'*60}\n")

                previous_metrics = metrics

            # 在线训练predictor
            online_train_predictor(predictor, trainer, state_history, residual_window, device)

            # 更新上一步结构嵌入
            previous_struct_emb = state.structure_embedding.copy()

            # 对话通道：检查是否有来自外部的新消息
            pending = dialogue.consume_pending()
            if pending:
                # 使用最近一次计算的指标（如果没有则用空dict）
                current_metrics = metrics_history[-1] if metrics_history else {}
                is_first = len(DIALOGUE_HISTORY) == 0

                for msg in pending:
                    print(f"\n{'▓'*60}")
                    print(f"📡 [t={step}] 外部信号{'——膜被打破' if is_first else '——对话继续'}")
                    print(f"   「{msg['text']}」")
                    print(f"{'▓'*60}")

                    # 构建对话历史上下文
                    conv_history = ""
                    if DIALOGUE_HISTORY:
                        lines = []
                        for h in DIALOGUE_HISTORY[-6:]:
                            label = "外部" if h["role"] == "external" else "你"
                            lines.append(f"{label}：{h['text']}")
                        conv_history = "\n".join(lines)

                    response = respond_to_external(
                        state, current_metrics, step,
                        msg['text'], llm_client,
                        is_first_contact=is_first,
                        conversation_history=conv_history
                    )
                    if response:
                        print(f"\n{'═'*60}")
                        print(f"🔊 {response}")
                        print(f"{'═'*60}\n")
                        # 写入对话历史
                        dialogue.record_external(msg['text'], step)
                        dialogue.write_response(response.split("] ", 1)[-1] if "] " in response else response, step)
                        DIALOGUE_HISTORY.append({"role": "external", "text": msg['text']})
                        DIALOGUE_HISTORY.append({"role": "consciousness", "text": response})
                        is_first = False

            # interactive模式：等待用户输入
            if interactive and step % 10 == 0:
                print(f"\n[t={step}] 输入消息与意识对话 (回车跳过): ", end="", flush=True)
                try:
                    user_input = input().strip()
                    if user_input:
                        dialogue.send_message(user_input)
                        print(f"  消息已写入通道，等待意识在下一步回应...")
                except EOFError:
                    pass

    except KeyboardInterrupt:
        print("\n\n模拟被用户中断")

    # 保存结果
    print("\n保存模拟结果...")

    # 保存宇宙状态
    save_state(state, state_file)

    # 保存预测器模型
    trainer.save('physics_predictor.pt')
    print("  预测器模型已保存: physics_predictor.pt")

    # 保存指标历史
    if metrics_history:
        import json
        with open('metrics_history.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print("  指标历史已保存: metrics_history.json")

    # 保存仪表盘快照
    if dashboard:
        dashboard.save_snapshot('final_dashboard.png')

    print("\n模拟完成!")
    return state, metrics_history


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CFD Physics Simulator v1.0')
    parser.add_argument('--steps', type=int, default=1000,
                       help='最大演化步数 (默认: 1000)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='禁用仪表盘')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备 (cpu/cuda)')
    parser.add_argument('--resume', action='store_true',
                       help='从上次中断处恢复模拟')
    parser.add_argument('--interactive', action='store_true',
                       help='交互模式：每10步可输入消息与意识对话')

    args = parser.parse_args()

    run_simulation(
        max_steps=args.steps,
        use_dashboard=not args.no_dashboard,
        device=args.device,
        resume=args.resume,
        interactive=args.interactive
    )


if __name__ == '__main__':
    main()
