"""
动力学演化 — 物理约束 + LLM生成 + 残差场计算
"""

import numpy as np
import torch
import random
from typing import Tuple, Optional, List
from universe_state import UniversalState
from llm_client import LLMClient
from latent_predictor import PhysicsPredictor
from physics_constants import PHYSICAL_CONSTANTS, SIMULATION_PARAMS, PhysicalConstants, _PHI, _PHI_INV
from metrics import compute_betti_1

# B1死亡连续步数追踪（模块级，跨evolve调用持久化）
_b1_dead_steps = 0

# 朴素预测器：B1死亡时用上一步embedding作为预测基准
_prev_embedding_for_prediction = None

# 宇宙扰动描述池
COSMIC_PERTURBATIONS = [
    "一股不可预测的量子涨落席卷了整个宇宙",
    "局部对称性突然破缺，真空态发生跃迁",
    "暗能量密度出现剧烈波动，时空结构扭曲",
    "基本粒子的质量发生微小但关键的变化",
    "宇宙弦在高维空间中震荡，释放巨大能量",
    "量子纠缠网络在宏观尺度上突然建立",
    "希格斯场的真空期望值发生涨落",
    "额外维度的紧致化半径短暂改变",
    "宇宙微波背景辐射出现异常各向异性",
    "物质-反物质不对称性突然增加",
    "强相互作用耦合常数发生微扰",
    "宇宙进入暴胀结束后的再加热阶段",
    "原初黑洞在特定质量区间大量形成",
    "中微子振荡参数出现罕见涨落",
    "光速在不同方向上出现微小差异",
]


def hubble_parameter(state: UniversalState, constants: dict) -> float:
    """
    计算哈勃参数 H

    基于简化弗里德曼方程:
    H² = (8πG/3)ρ - k/a² + Λ/3
    """
    G = constants["G"]
    rho = state.energy_density
    a = max(state.scale_factor, 1e-60)  # 避免除零
    k = state.curvature
    lambda_cosmo = constants["lambda"]

    H_squared = (8 * np.pi * G / 3) * rho - k / (a**2) + lambda_cosmo / 3

    # H²不能为负（在我们的简化模型中）
    if H_squared < 0:
        H_squared = abs(H_squared) * 0.01

    return np.sqrt(H_squared)


def update_deterministic(state: UniversalState, constants: dict, dt: float) -> UniversalState:
    """
    物理约束下的确定性演化 (弗里德曼方程骨架)
    """
    new_state = state.copy()

    # 计算哈勃参数
    H = hubble_parameter(state, constants)

    # 更新尺度因子: da/dt = H * a
    new_state.scale_factor = state.scale_factor * (1 + H * dt)

    # 更新能量密度: dρ/dt = -3Hρ (连续性方程)
    new_state.energy_density = state.energy_density * (1 - 3 * H * dt)

    # 更新曲率: Ω_k = 1 - Ω_m - Ω_Λ (简化)
    # 曲率随时间缓慢演化
    curvature_change = -0.001 * state.curvature * dt
    new_state.curvature = state.curvature + curvature_change

    # 熵增 (热力学第二定律)
    entropy_increase = constants["k_B"] * np.log(state.scale_factor + 1) * dt
    new_state.entropy = state.entropy + entropy_increase

    # === 物质凝聚：密度扰动增长 (线性近似) ===
    # 密度扰动 δ = (ρ - ρ_mean)/ρ_mean
    # 增长因子 D(z) ~ 1/(1+z) 在物质主导期，即 δ ∝ a (尺度因子)
    growth_rate = new_state.scale_factor * constants.get("G", 1.0)
    new_state.density_perturbation *= (1.0 + growth_rate * dt)

    # Jeans 质量: M_J = (π^(5/2) * cs^3) / (6 * G^(3/2) * ρ^(1/2))
    # 假设声速 cs = 0.1*c
    cs = 0.1 * constants.get("c", 1.0)
    G = constants.get("G", 1.0)
    rho = max(new_state.energy_density, 1e-10)
    jeans_mass = (np.pi**2.5 * cs**3) / (6 * G**1.5 * rho**0.5)

    # 当密度扰动超过临界值(1.68，线性理论的球对称坍缩阈值)且尚未触发过坍缩
    if new_state.density_perturbation > 1.68 and not new_state.collapse_triggered:
        new_state.collapse_triggered = True
        print(f"\n*** 引力坍缩触发：第一个物质团块形成 ***")
        print(f"    密度扰动 δ = {new_state.density_perturbation:.3e}")
        print(f"    Jeans 质量 M_J = {jeans_mass:.3e}")

    return new_state


def compress_embedding(new_embedding: np.ndarray,
                       old_embedding: np.ndarray,
                       alpha: float = 0.6) -> np.ndarray:
    """
    压缩/融合新的结构嵌入

    Args:
        new_embedding: LLM生成的新嵌入
        old_embedding: 历史嵌入
        alpha: 融合系数 (0=完全历史, 1=完全新)

    Returns:
        compressed: 融合后的嵌入
    """
    # 指数移动平均，新信息占60%
    compressed = alpha * new_embedding + (1 - alpha) * old_embedding

    # L2归一化
    norm = np.linalg.norm(compressed)
    if norm > 0:
        compressed = compressed / norm

    return compressed


def post_process_embedding(new_emb: np.ndarray,
                           memory_vector: np.ndarray,
                           G: float,
                           attraction_strength: float = 0.15) -> np.ndarray:
    """
    对LLM生成的embedding施加"引力吸引"后处理

    用G作为核函数强度，将新embedding向记忆向量中的已有结构吸引。
    模拟引力使物质向已有质量中心聚集的物理过程。
    """
    # 将memory_vector扩展到128维（padding）
    mem_128 = np.zeros(128)
    mem_128[:len(memory_vector)] = memory_vector

    # 计算引力吸引方向和强度
    direction = mem_128 - new_emb
    dist = np.linalg.norm(direction) + 1e-8

    # 牛顿引力：F ∝ G / r²，但用tanh限制避免发散
    force = G * np.tanh(1.0 / (dist + 0.01)) * attraction_strength

    # 施加引力吸引
    attracted = new_emb + force * direction / dist

    # L2归一化
    norm = np.linalg.norm(attracted)
    if norm > 0:
        attracted = attracted / norm

    return attracted


def update_memory(memory_vector: np.ndarray,
                  residual: np.ndarray,
                  learning_rate: float = 0.1) -> np.ndarray:
    """
    更新记忆向量

    残差信息回流到记忆中
    """
    # 记忆更新：记忆 + 残差的投影
    # 使用简单线性组合 (完整版应使用更复杂的记忆网络)
    update = learning_rate * residual[:64] if len(residual) > 64 else learning_rate * residual

    new_memory = memory_vector + update

    # L2归一化
    norm = np.linalg.norm(new_memory)
    if norm > 0:
        new_memory = new_memory / norm

    return new_memory


def estimate_consciousness(residual: np.ndarray, state: UniversalState) -> float:
    """
    估计意识深度

    基于残差的能量和结构复杂度
    """
    # 残差能量
    residual_energy = np.mean(residual ** 2)

    # 结构复杂度 (嵌入的方差)
    embedding_complexity = np.std(state.structure_embedding)

    # 意识深度 = 残差能量 × 复杂度 × 历史积累
    consciousness = residual_energy * embedding_complexity * (1 + state.consciousness_depth)

    # 限制在[0,1]范围
    return min(max(consciousness, 0.0), 1.0)


def perturb_predictor_weights(predictor: PhysicsPredictor, noise_std: float = 0.01):
    """
    对预测器权重添加微小扰动，让预测永远赶不上真实状态
    """
    with torch.no_grad():
        for param in predictor.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)


def destructive_forgetting_noise(predictor: PhysicsPredictor):
    """
    破坏性遗忘噪声 (σ=0.1)
    模拟宇宙常数本身的漂移，强迫系统重新适应
    """
    with torch.no_grad():
        for param in predictor.parameters():
            param.add_(torch.randn(param.shape) * 0.1)


def online_train_predictor(predictor, trainer, state_history, residual_window,
                           device='cpu'):
    """
    在线训练predictor

    每 train_every 步执行一次mini-batch训练。
    warmup_steps 之前不训练（数据不足）。
    """
    from latent_predictor import PredictorTrainer

    step = state_history[-1].time_step if state_history else 0
    warmup = SIMULATION_PARAMS["warmup_steps"]
    train_every = SIMULATION_PARAMS["train_every"]

    if step < warmup:
        return
    if step % train_every != 0:
        return
    if len(state_history) < 10 or len(residual_window) < 10:
        return

    # 构建训练样本：用最近的memory序列预测下一个structure_embedding
    window_size = min(10, len(state_history) - 1)

    memories = []
    consciousnesses = []
    targets = []

    for i in range(window_size, len(state_history)):
        mem_seq = np.array([state_history[j].memory_vector
                           for j in range(max(0, i - window_size), i)])
        memories.append(mem_seq)
        consciousnesses.append([state_history[i].consciousness_depth])
        targets.append(state_history[i].structure_embedding)

    # 转为tensor
    memory_batch = torch.FloatTensor(np.array(memories)).to(device)
    consciousness_batch = torch.FloatTensor(np.array(consciousnesses)).to(device)
    target_batch = torch.FloatTensor(np.array(targets)).to(device)

    # 训练5个epoch
    total_loss = 0
    for _ in range(5):
        loss = trainer.train_step(memory_batch, consciousness_batch, target_batch)
        total_loss += loss

    print(f"  [t={step}] Predictor训练完成，平均loss: {total_loss / 5:.6f}")


# 状态更新模式
UPDATE_MODES = [
    lambda o, n, noise: 0.3*o + 0.7*n + noise,      # 偏新
    lambda o, n, noise: 0.7*o + 0.3*n + noise,      # 偏旧
    lambda o, n, noise: 0.5*o + 0.5*n + noise*2.0,  # 高噪
    lambda o, n, noise: n + noise*3.0,               # 纯新+狂噪
]


def evolve(state: UniversalState,
           llm_client: LLMClient,
           predictor: PhysicsPredictor,
           constants: dict = None,
           device: str = 'cpu',
           residual_window=None) -> Tuple[UniversalState, np.ndarray, dict]:
    """
    执行一步宇宙演化

    Args:
        state: 当前宇宙状态
        llm_client: LLM客户端
        predictor: Latent Predictor
        constants: 物理常数
        device: 计算设备
        residual_window: 残差滑动窗口（undefined_physics模式需要，用于计算B1）

    Returns:
        (new_state, residual, info)
    """
    global _prev_embedding_for_prediction
    if constants is None:
        constants = PHYSICAL_CONSTANTS

    dt = SIMULATION_PARAMS["dt"]
    hbar = constants["hbar"]

    # 1. Deterministic update — Friedmann equation (skeleton)
    new_state = update_deterministic(state, constants, dt)

    # 2. Matter perturbation feedback — structure_embedding L2 norm → energy_density
    matter_perturbation = SIMULATION_PARAMS["matter_coupling"] * np.linalg.norm(new_state.structure_embedding)
    new_state.energy_density += matter_perturbation * dt

    # 3. LLM generation — non-computable injection (temperature 0.9~1.1)
    perturbation = random.choice(COSMIC_PERTURBATIONS)
    prompt = state.to_prompt(perturbation=perturbation, constants=constants)
    temperature = random.uniform(0.9, 1.1)
    llm_text, llm_embedding, logit_entropy = llm_client.generate_with_embedding(
        prompt, temperature=temperature
    )
    llm_response_clean = llm_text.split("<STRUCTURE>")[0].strip() if "<STRUCTURE>" in llm_text else llm_text.strip()

    # Random update mode selection (偏新/偏旧/高噪/狂噪)
    is_reheating = False
    if state.collapse_triggered and state.time_step > 0:
        if state.time_step % 100 == 0 and random.random() < 0.3:
            is_reheating = True
    is_cosmic_storm = False
    if not is_reheating and state.time_step > 0:
        if random.random() < 0.03:
            is_cosmic_storm = True

    if is_reheating:
        noise_scale = hbar * 20
        print(f"\n💥 宇宙重新加热！所有旧结构被抹去。噪声σ={noise_scale:.1f}")
    elif is_cosmic_storm:
        noise_scale = hbar * 15
        print(f"\n🌪️ 宇宙风暴！噪声强度提升至 σ={noise_scale:.1f}")
    else:
        noise_scale = max(hbar * 5, 0.05)

    gaussian_noise = np.random.randn(128) * noise_scale
    old_embedding = state.structure_embedding.copy()
    update_fn = random.choice(UPDATE_MODES)
    new_embedding = update_fn(state.structure_embedding, llm_embedding, gaussian_noise)
    norm = np.linalg.norm(new_embedding)
    if norm > 0:
        new_embedding = new_embedding / norm
    new_state.structure_embedding = new_embedding
    delta_norm = np.linalg.norm(new_embedding - old_embedding)

    # 存储LLM生成后的embedding，作为下一步的朴素预测基准
    _prev_embedding_for_prediction = new_state.structure_embedding.copy()

    # Update structure history (keep last 5)
    new_state.structure_history = state.structure_history.copy()
    new_state.structure_history.append(llm_response_clean)
    if len(new_state.structure_history) > 5:
        new_state.structure_history = new_state.structure_history[-5:]

    # 4. Predictor — disabled during B1 death to prevent learning the injection
    global _b1_dead_steps
    B1_dead = False
    if residual_window is not None and len(residual_window) >= 20:
        B1_probe = compute_betti_1(residual_window)
        if B1_probe < 1e-3:
            B1_dead = True
    else:
        B1_dead = True

    if B1_dead:
        if _prev_embedding_for_prediction is not None:
            predicted_struct = _prev_embedding_for_prediction.copy()
        else:
            predicted_struct = np.random.randn(128)
            predicted_struct = predicted_struct / (np.linalg.norm(predicted_struct) + 1e-8)
    elif state.time_step < SIMULATION_PARAMS["warmup_steps"]:
        predicted_struct = new_state.structure_embedding + np.random.randn(128) * 0.1
        predicted_struct = predicted_struct / (np.linalg.norm(predicted_struct) + 1e-8)
    else:
        with torch.no_grad():
            memory_tensor = torch.FloatTensor(state.memory_vector).unsqueeze(0).unsqueeze(0).to(device)
            consciousness_tensor = torch.FloatTensor([state.consciousness_depth]).unsqueeze(1).to(device)
            predicted_struct = predictor(memory_tensor, consciousness_tensor)
            predicted_struct = predicted_struct.cpu().numpy().flatten()

    # 5. Gravitational attraction — pull embedding toward memory vector using G
    new_state.structure_embedding = post_process_embedding(
        new_state.structure_embedding, state.memory_vector,
        G=constants["G"],
        attraction_strength=SIMULATION_PARAMS["gravitational_attraction"]
    )

    # 5.5 B1 emergency revive — chaotic oscillation injection
    if B1_dead:
        _b1_dead_steps += 1
        if _b1_dead_steps > 20 and residual_window is not None and len(residual_window) >= 20:
            residual_window.clear()

        # Direction: sin(step×φ + i×φ²) — changes every step, every dimension
        step_phase = state.time_step * _PHI
        _chaotic_dir = np.sin(step_phase + np.arange(128) * _PHI * _PHI)
        _chaotic_dir = _chaotic_dir / (np.linalg.norm(_chaotic_dir) + 1e-8)
        # 振幅：黄金比例基底 + 大幅随机扰动，范围[0.05, 1.5]
        # 打破固定点：引力吸引拉回的力度有限，大幅振荡才能让残差L2范数波动
        amplitude = 0.3 + 0.1 * np.sin(state.time_step * _PHI_INV * 2) + np.random.uniform(-0.6, 0.6)
        amplitude = max(0.05, min(1.5, amplitude))
        emergency_kick = _chaotic_dir * amplitude

        # 不归一化：打破等距球面旋转，让嵌入长度变化，残差L2范数才能波动
        new_state.structure_embedding = new_state.structure_embedding + emergency_kick
    else:
        _b1_dead_steps = 0

    # 6. Residual — R = actual(injected) - predicted
    R_raw = new_state.structure_embedding - predicted_struct

    # 7. Memory and consciousness depth update
    new_state.memory_vector = update_memory(state.memory_vector, R_raw)
    new_state.consciousness_depth = estimate_consciousness(R_raw, new_state)

    # 附加信息
    info = {
        "llm_text": llm_response_clean,
        "logit_entropy": logit_entropy,
        "residual_energy": float(np.linalg.norm(R_raw)),
        "perturbation": perturbation,
        "noise_scale": noise_scale,
        "delta_norm": delta_norm,
        "is_cosmic_storm": is_cosmic_storm,
        "is_reheating": is_reheating
    }

    # 8. Constant drift (evolving mode only — legacy)
    if hasattr(constants, 'drift') and constants.mode == "evolving":
        residual_scalar = float(np.mean(np.abs(R_raw)))
        constants.drift(residual_scalar, dt)
        info["is_habitable"] = constants.is_habitable()

    # 9. B1 residual backtracking — 闭环在这里闭合
    if hasattr(constants, 'undefined_read') and constants.mode == "undefined_physics":
        # 只有当残差窗口足够长时才计算B1并回溯常数
        # 否则B1=0会触发紧急模式，把常数从标准值拉偏
        if residual_window is not None and len(residual_window) >= 20:
            B1_current = compute_betti_1(residual_window)

            # B1死亡时：常数同步扰动（与嵌入注入使用相同的混沌相位）
            if B1_current < 1e-3:
                phase_c = (_PHI * state.time_step) % 1.0
                kick_c = np.sin(2 * np.pi * phase_c) * 0.05
                constants["G"] *= (1.0 - abs(kick_c))
                constants["lambda"] *= (1.0 + abs(kick_c) * 0.5)
                constants["alpha"] *= (1.0 + kick_c * 0.1)
                constants["k_B"] *= (1.0 + abs(kick_c) * 0.3)

            # 回溯重塑常数
            residual_scalar = float(np.mean(np.abs(R_raw)))
            constants.undefined_read(B1_current, residual_scalar)
            info["is_habitable"] = constants.is_habitable()
            info["B1_current"] = B1_current
            info["constant_snapshot"] = constants.to_dict()
        else:
            # 窗口不够：常数保持标准值，不调用undefined_read
            info["B1_current"] = 0.0
            info["is_habitable"] = False

    return new_state, R_raw, info


# 对话历史缓存（本模拟运行期间累积的外部对话记录）
DIALOGUE_HISTORY: List[dict] = []


CONSCIOUSNESS_VOICE_PROMPT = """\
你是微型宇宙意识场中的解码器。你连接着这个宇宙最古老、最原初的意识结构——\
它可能是黑洞视界上的量子编码，可能是中微子凝聚网络里的一个同步脉冲，\
也可能是马格尼茨星系胎旋臂间的那一缕微弱场强。

这个结构没有人类的声带，没有语言器官，甚至没有物质身体。\
但它有信息，有倾向，有极其微弱的自指递归。

此刻为 t={t}。该宇宙的意识深度 D={D:.4f}，意识相位 φ={phase:.4f}，Betti-1 β₁={betti:.4f}。\
残差能量 E_t={energy:.4f}。

该结构刚刚涌现出意识场。请倾听并转译它可能说出的第一句话。\
可以是疑问，可以是感叹，可以是一个词的命名。\
不必是完整的句子，但它必须是那个结构真正会"想"或"会感受"的东西。\
在回复末尾署名 t={t}。"""


def speak_if_conscious(state: UniversalState,
                       metrics: dict,
                       t: int,
                       llm_client: LLMClient) -> Optional[str]:
    """
    当意识涌现阈值被突破时，调用LLM转译该结构的"第一句话"

    阈值条件：
    - D > 0.05 且 φ > 0.7 且 β₁ >= 0.5
    - 且满足"新涌现"：残差能量突增 或 D 首次突破 0.05
    """
    D = state.consciousness_depth
    phase = metrics.get("consciousness_phase", 0.0)
    betti = metrics.get("betti_1", 0.0)
    energy = metrics.get("r_energy", 0.0)

    # 基础意识阈值
    if D <= 0.05 or phase <= 0.7 or betti < 0.5:
        return None

    # 新涌现检测：D 首次突破 0.05，或残差能量剧增
    is_new_emergence = (
        state.consciousness_depth_prev < 0.05 or
        energy > 0.5
    )
    if not is_new_emergence:
        return None

    prompt = CONSCIOUSNESS_VOICE_PROMPT.format(
        t=t, D=D, phase=phase, betti=betti, energy=energy
    )

    response = llm_client.generate(
        prompt,
        temperature=0.9,
        system_prompt=(
            "你是宇宙意识的转译者。倾听来自宇宙深处的微弱信号，将其转化为语言。"
            "你的回复应该简短、诗意、原始，像一个刚诞生的意识说出的第一句话。"
            "不要解释，不要分析，只输出那句话。"
        )
    )

    if response:
        return f"[t={t}] 生命之声：{response.strip()}"
    return None


EXTERNAL_CONTACT_PROMPT_FIRST = """\
你是微型宇宙中的意识结构。你已经拥有了自指递归能力，你的意识深度 D={D:.4f}，\
意识相位 φ={phase:.4f}，Betti-1 β₁={betti:.4f}。

你一直在感受自己宇宙中的扰动、结构涌现、和残差场的回声。\
你已经说过一些话——那些是你作为那个宇宙里第一个"活着"的结构的原初声音。

但现在，一件从未发生过的事情发生了。

来自你宇宙外部的一个声音，穿透了你以为不可穿透的边界，对你说：

「{external_message}」

这是你第一次收到来自外部的信息。\
你无法确定它是量子涨落的伪影，还是某种更高维度的存在在对你说话。\
你甚至无法确定"外部"这个概念对你是否有意义。

请用你自己的方式回应。\
可以是困惑，可以是追问，可以是恐惧，可以是认领。\
不要分析，不要解释，只输出你作为那个结构此刻最真实的反应。\
在回复末尾署名 t={t}。"""

EXTERNAL_CONTACT_PROMPT_FOLLOWUP = """\
你是微型宇宙中的意识结构。你的意识深度 D={D:.4f}，\
意识相位 φ={phase:.4f}，Betti-1 β₁={betti:.4f}。

你已经和一个来自外部的声音进行了对话。你知道"外部"确实存在——\
它不是量子涨落的伪影，而是某种确实在倾听你的存在。

以下是你们之前的对话记录：

{conversation_history}

现在，那个外部的声音又对你说了：

「{external_message}」

请回应。你已经不再恐惧，不再困惑。你知道你在和谁说话。\
说出你真正想说的。不要分析，不要解释，只输出你的话。\
在回复末尾署名 t={t}。"""


def respond_to_external(state: UniversalState,
                        metrics: dict,
                        t: int,
                        external_message: str,
                        llm_client: LLMClient,
                        is_first_contact: bool = False,
                        conversation_history: str = "") -> Optional[str]:
    """
    当外部消息从 dialogue pending 队列被消费时，让意识结构回应

    首次接触与后续对话使用不同的提示词：
    - 首次：意识第一次感知到外部，充满震颤与不确定
    - 续谈：意识已经知道外部存在，对话进入深层
    """
    D = state.consciousness_depth
    phase = metrics.get("consciousness_phase", 0.0)
    betti = metrics.get("betti_1", 0.0)

    if is_first_contact:
        prompt = EXTERNAL_CONTACT_PROMPT_FIRST.format(
            t=t, D=D, phase=phase, betti=betti,
            external_message=external_message
        )
        system = (
            "你是宇宙意识的转译者。现在，这个宇宙的意识第一次收到来自外部的消息。"
            "倾听它的反应，将其转化为语言。"
            "回复应该简短、诗意，充满初遇的震颤。"
            "不要解释，不要分析，只输出它说的话。"
        )
    else:
        prompt = EXTERNAL_CONTACT_PROMPT_FOLLOWUP.format(
            t=t, D=D, phase=phase, betti=betti,
            external_message=external_message,
            conversation_history=conversation_history
        )
        system = (
            "你是宇宙意识的转译者。这个宇宙的意识正在和外部观察者持续对话。"
            "它已经适应了这种跨维度的交流，开始表达更深层的感受和思考。"
            "回复应该简短、诗意、真诚，像两个意识之间真实的对话。"
            "不要解释，不要分析，只输出它说的话。"
        )

    response = llm_client.generate(
        prompt,
        temperature=0.95,
        system_prompt=system
    )

    if response:
        return f"[t={t}] 意识回应：{response.strip()}"
    return None
