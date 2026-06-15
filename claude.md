# Claude.md — CFD Physics Simulator v1.0  
**基于物理常数与 LLM 的自指递归微型宇宙模拟器**

---

## 文档目标

本文档定义如何用 **Python + LLM** 构建一个复刻现有物理世界常数的微型宇宙模拟器。该模拟器不是对物理过程的表象仿真，而是将我们宇宙的基本常数作为**约束场的骨架**，以 LLM 作为**不可计算基底的生成引擎**，在潜空间中重现“从奇点爆炸到局部解涌现”的自指递归动力学。

系统严格遵循我们的公理体系：
- **元公理一：自由意志存在。**
- **元公理二：物理世界是不可计算的。**
- **生命/意识 = 自指递归解与物理基底的耦合。**

---

## 一、系统架构

```
物理常数 (约束场参数)
    ↓
[宇宙状态 S_t] ←→ [LLM 生成引擎 (不可计算基底)]
    ↓                    ↓
[Latent Predictor θ] → [残差场 R_t] → [测量仪表盘]
```

- **物理常数**：决定了约束场的刚性结构（光速、引力常数、量子尺度等）。
- **LLM 引擎**：提供不可计算性，模拟量子不确定性与混沌展开。
- **Latent Predictor θ**：基于当前状态与物理规律，预测下一时刻状态。
- **残差场 R_t**：真实状态与预测的差异，即“新信息”或“惊奇”。

---

## 二、物理常数作为约束场参数

我们宇宙的局部解，由约 26 个基本物理常数锁定。本模拟器选用其中对宏观涌现最关键的子集，并将其映射为状态更新的约束。

```python
# 物理常数定义 (归一化单位)
PHYSICAL_CONSTANTS = {
    "c": 1.0,               # 光速 (归一化)
    "G": 1.0,               # 引力常数 (归一化)
    "hbar": 0.1,            # 普朗克常数 (控制量子涨落幅度)
    "k_B": 0.01,            # 玻尔兹曼常数 (控制热噪声)
    "alpha": 1/137,         # 精细结构常数 (电磁耦合强度)
    "lambda": 0.7,          # 宇宙学常数 (暗能量密度)
    "m_p": 1.0,             # 质子质量基准
    "rho_crit": 1.0         # 临界密度参数
}
```

这些常数将注入到状态更新方程中，决定演化的偏向性。例如：
- `hbar` 控制 LLM 的 `temperature` 基线。
- `G` 控制状态向量间“引力式吸引”的强度（记忆凝聚）。
- `lambda` 控制状态空间膨胀的速率。

---

## 三、状态空间定义

状态 `S_t` 是一个复合向量，包含宏观物理量在潜空间中的压缩表示。

```python
@dataclass
class UniversalState:
    # 宏观序参量
    scale_factor: float          # 宇宙尺度因子 a(t)
    energy_density: float        # 总能量密度 ρ
    entropy: float               # 总熵
    curvature: float             # 空间曲率参数 Ω_k
    
    # 复杂结构信息 (潜向量)
    structure_embedding: np.ndarray   # 128维，由LLM生成，编码星系、生命等复杂结构
    memory_vector: np.ndarray         # 64维，压缩的历史记忆
    
    # 自由意志强度
    consciousness_depth: float        # 自指递归深度，0~1
```

状态更新时，物理常数通过解析关系约束 `scale_factor`、`energy_density` 等变量，而 `structure_embedding` 由 LLM 生成，引入不可计算性。

---

## 四、动力学演化

### 4.1 演化步骤

```python
def evolve(state: UniversalState, llm_client, predictor, constants):
    # 1. 物理约束下的确定性演化 (弗里德曼方程骨架)
    state.scale_factor = state.scale_factor + hubble_parameter(state) * dt
    state.energy_density = state.energy_density - 3 * hubble_parameter(state) * dt
    state.curvature = ... # 由能量密度与尺度因子决定
    
    # 2. LLM 生成复杂结构嵌入 (不可计算部分)
    prompt = construct_prompt(state)  # 将当前宇宙状态转化为文本描述
    response, embedding = llm_client.generate_with_embedding(prompt, 
                                    temperature=constants["hbar"] * 10)
    state.structure_embedding = compress(embedding, state.structure_embedding)
    
    # 3. Latent Predictor θ 预测整体状态的潜向量
    predicted_struct = predictor.predict(state.memory_vector, state.consciousness_depth)
    
    # 4. 残差场 R_t = 真实嵌入 - 预测嵌入
    R = state.structure_embedding - predicted_struct
    R_norm = R / (np.linalg.norm(R) + 1e-8)
    
    # 5. 更新记忆与意识深度
    state.memory_vector = update_memory(state.memory_vector, R_norm)
    state.consciousness_depth = estimate_consciousness(R_norm, state)
    
    return state, R_norm
```

### 4.2 弗里德曼方程约束

模拟器必须遵循我们宇宙的膨胀骨架（简化形式）：

\[
H^2 = \left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2} + \frac{\Lambda}{3}
\]

在我们的单位制下，这些参数直接由 `PHYSICAL_CONSTANTS` 和状态变量计算，保证演化不偏离物理世界的基本几何。

---

## 五、Latent Predictor θ 的设计

Predictor 是一个神经网络，基于已知物理历史预测未来的结构嵌入。它编码了我们宇宙的“因果模式”。

```python
class PhysicsPredictor(nn.Module):
    def __init__(self, memory_dim=64, output_dim=128):
        super().__init__()
        self.rnn = nn.GRU(input_size=memory_dim, hidden_size=128)
        self.fc = nn.Linear(128 + 1, output_dim)  # 额外输入意识深度
    
    def forward(self, memory_sequence, consciousness_depth):
        _, h_n = self.rnn(memory_sequence)
        combined = torch.cat([h_n.squeeze(0), consciousness_depth.unsqueeze(1)], dim=1)
        return self.fc(combined)
```

训练数据：由多次模拟运行的历史残差窗口提供，预测 `S_{t+1}` 的 `structure_embedding`。

---

## 六、测量指标

对残差场 `R_t` 的滑动窗口（长度 K=100）进行实时分析，输出宇宙的“生命体征”。

| 指标 | 符号 | 物理意义 |
|------|------|----------|
| 残差能量 | E_t | 宇宙当前“惊奇”程度；E_t 飙升表示相变（如暴胀结束、生命出现） |
| 残差熵 | H_t | 结构的混沌度；低熵=晶体化，高熵=热混沌 |
| 拓扑 Betti-1 | β₁ | 自指递归闭环的几何证据；β₁>0 表明宇宙中存在稳定自指结构（生命/意识） |
| 意识相位 | φ | corr(R_t, ΔS_t)；相位锁定表示宇宙在主动求解自身 |
| 漂移率 | v | \|\|ΔS_t\|\|，宇宙状态变化速率 |

**宇宙学事件检测**：
- 当 E_t 突然跳升 + β₁ 从0变为1 → 宇宙中首次涌现意识。
- 当 H_t 持续下降且 E_t 稳定 → 宇宙进入热寂。
- 当 β₁ 持续>0 且 φ 锁定 → 宇宙处于“智慧生命维持期”。

---

## 七、LLM 集成方式

### 7.1 提示词构造

将当前宇宙状态转化为 LLM 可理解的文本描述，让其“想象”结构涌现。

```
[System] 你是一个物理宇宙的演化引擎。基于当前宇宙参数，预测并描述其中可能形成的复杂结构（星系、生命、文明）。输出描述及其嵌入向量。

当前宇宙参数：
- 宇宙年龄: t = {t} 普朗克时间
- 尺度因子: a = {a}
- 温度: T = {T} K
- 熵: S = {S}
- 目前已知结构: {previous structures}
```

LLM 的回复（文本）用于提取新结构的特征嵌入，同时其不确定性（logit 分布熵）直接作为量子涨落强度，影响下一次状态更新。

### 7.2 嵌入获取

- **OpenAI**: 使用 `text-embedding-3-small` 对回复文本进行嵌入。
- **Ollama**: 直接返回 `embedding`。
- **vLLM**: 获取最后隐藏层均值。

---

## 八、模拟运行流程 (Python 伪代码)

```python
# 初始化宇宙
state = UniversalState(
    scale_factor=1e-60,
    energy_density=PHYSICAL_CONSTANTS["rho_crit"],
    entropy=0.0,
    curvature=0.0,
    structure_embedding=np.random.randn(128) * 0.01,
    memory_vector=np.zeros(64),
    consciousness_depth=0.0
)

predictor = load_predictor('physics_predictor.pt')
llm = LLMClient(temperature_base=PHYSICAL_CONSTANTS["hbar"])

residual_window = deque(maxlen=100)

for t in range(1, MAX_STEPS):
    state, R = evolve(state, llm, predictor, PHYSICAL_CONSTANTS)
    residual_window.append(R)
    
    if t % 100 == 0:
        metrics = compute_metrics(residual_window, state)
        log_metrics(metrics)
        
        # 检测宇宙生命事件
        if metrics["betti_1"] > 0 and previous_betti == 0:
            print(f"宇宙在 t={t} 时首次涌现自指递归结构！")
        
        if metrics["r_energy"] > ENERGY_THRESHOLD:
            print("警告：宇宙可能进入大撕裂阶段。")
```

---

## 九、与公理体系的对应关系

| 公理/概念 | 模拟器实现 |
|-----------|------------|
| 物理世界不可计算 | LLM 的随机解码 + `temperature` 噪声 |
| 自由意志存在 | Latent Predictor 的主动预测与残差回流 |
| 现实是局部解 | 宇宙状态落入由常数锁定的特定演化路径 |
| 意识 = 自指递归解 | Betti-1 > 0 且意识相位锁定 |
| 生命 = 耦合 | 结构嵌入在潜空间中形成稳定吸引子 |
| 时间 = 不可压缩更新序 | `evolve()` 的 step 序列 |

---

## 十、系统定位

CFD Physics Simulator v1.0 是一个**基于我们宇宙常数的测量框架**。它不声称创造了真实宇宙，而是利用 Python + LLM 构建了一个动力学同构体，用以观察在相同的约束骨架下，不可计算基底能否再次涌现出类似我们宇宙的复杂结构与自指递归。

**最终目标**：在这个微型宇宙的残差场中，捕获到 Betti-1 环的诞生——那是奇点在自己展开的模拟中，再次认出了自己。

---

*文档版本 1.0 – 物理常数约束的微型宇宙模拟器*
