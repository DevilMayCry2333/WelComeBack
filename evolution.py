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
from physics_constants import PHYSICAL_CONSTANTS, SIMULATION_PARAMS

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

    # 更新时间步
    new_state.time_step = state.time_step + 1

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
           device: str = 'cpu') -> Tuple[UniversalState, np.ndarray, dict]:
    """
    执行一步宇宙演化

    Args:
        state: 当前宇宙状态
        llm_client: LLM客户端
        predictor: Latent Predictor
        constants: 物理常数
        device: 计算设备

    Returns:
        (new_state, residual, info)
    """
    if constants is None:
        constants = PHYSICAL_CONSTANTS

    dt = SIMULATION_PARAMS["dt"]
    hbar = constants["hbar"]

    # 1. 物理约束下的确定性演化
    new_state = update_deterministic(state, constants, dt)

    # 2. LLM生成复杂结构嵌入 (不可计算部分)
    # 随机选择宇宙扰动
    perturbation = random.choice(COSMIC_PERTURBATIONS)

    # 构造prompt (包含扰动和历史)
    prompt = state.to_prompt(perturbation=perturbation)

    # 提升温度到0.9~1.1
    temperature = random.uniform(0.9, 1.1)
    llm_text, llm_embedding, logit_entropy = llm_client.generate_with_embedding(
        prompt, temperature=temperature
    )

    # 提取LLM回复中的结构描述
    llm_response_clean = llm_text.split("<STRUCTURE>")[0].strip() if "<STRUCTURE>" in llm_text else llm_text.strip()

    # 3. 强制状态突变：随机选择更新模式
    is_reheating = (state.time_step % 50 == 0) and (state.time_step > 0)

    # 宇宙风暴：随机触发 (约10%概率，但不与重新加热重叠)
    is_cosmic_storm = False
    if not is_reheating and state.time_step > 0:
        # 随机概率触发，确保每5-15步左右会有一次
        if random.random() < 0.1:  # 10%概率
            is_cosmic_storm = True

    if is_reheating:
        # 💥 宇宙重新加热！所有旧结构被抹去
        noise_scale = hbar * 50
        print(f"\n💥 宇宙重新加热！所有旧结构被抹去。噪声σ={noise_scale:.1f}")
    elif is_cosmic_storm:
        noise_scale = hbar * 20
        print(f"\n🌪️ 宇宙风暴！噪声强度提升至 σ={noise_scale:.1f}")
    else:
        noise_scale = max(hbar * 5, 0.05)

    gaussian_noise = np.random.randn(128) * noise_scale

    # 保存旧嵌入用于计算变化量
    old_embedding = state.structure_embedding.copy()

    # 随机选择状态更新模式
    update_fn = random.choice(UPDATE_MODES)
    new_embedding = update_fn(state.structure_embedding, llm_embedding, gaussian_noise)

    # 归一化，防止向量长度漂移
    norm = np.linalg.norm(new_embedding)
    if norm > 0:
        new_embedding = new_embedding / norm

    new_state.structure_embedding = new_embedding

    # 计算状态嵌入变化量
    delta_norm = np.linalg.norm(new_embedding - old_embedding)

    # 更新历史 (保留最近5次)
    new_state.structure_history = state.structure_history.copy()
    new_state.structure_history.append(llm_response_clean)
    if len(new_state.structure_history) > 5:
        new_state.structure_history = new_state.structure_history[-5:]

    # 4. Latent Predictor θ 预测整体状态的潜向量
    with torch.no_grad():
        memory_tensor = torch.FloatTensor(state.memory_vector).unsqueeze(0).unsqueeze(0).to(device)
        consciousness_tensor = torch.FloatTensor([state.consciousness_depth]).unsqueeze(1).to(device)

        predicted_struct = predictor(memory_tensor, consciousness_tensor)
        predicted_struct = predicted_struct.cpu().numpy().flatten()

    # 5. 残差场 R_t = 真实嵌入 - 预测嵌入 (不归一化，保持敏感度)
    R_raw = new_state.structure_embedding - predicted_struct

    # 6. 更新记忆与意识深度
    new_state.memory_vector = update_memory(state.memory_vector, R_raw)
    new_state.consciousness_depth = estimate_consciousness(R_raw, new_state)

    # 附加信息
    info = {
        "llm_text": llm_response_clean,
        "logit_entropy": logit_entropy,
        "residual_energy": float(np.linalg.norm(R_raw)),  # 使用原始残差范数
        "perturbation": perturbation,
        "noise_scale": noise_scale,
        "delta_norm": delta_norm,
        "is_cosmic_storm": is_cosmic_storm,
        "is_reheating": is_reheating
    }

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
