"""
物理常数定义 — 基于我们宇宙的基本常数，归一化单位

PhysicalConstants 类支持四种模式：
- standard: 使用我们宇宙的标准常数
- habitable_random: 在可居住窗口内随机采样
- evolving: 从标准值开始，允许柔性常数随残差场漂移
- undefined_physics: 物理常数成为B1的因变量，常数永远在逼近但从不抵达
"""

import numpy as np

# 黄金比例连分数展开系数（用于生成无理数扰动序列）
_PHI = (1 + np.sqrt(5)) / 2  # 黄金比例 ≈ 1.6180339887...
_PHI_INV = 1.0 / _PHI  # 1/φ ≈ 0.6180339887... — B1的目标值


def _golden_ratio_continued_fraction(step: int) -> float:
    """
    黄金比例的连分数逼近序列：[1, 1, 2, 3, 5, 8, 13, ...]
    每一步返回一个无理数残余，永远不终止
    """
    if step <= 0:
        return 0.0
    # 连分数收敛子：F(n)/F(n+1) → 1/φ
    # 扰动量 = (-1)^n / (F(n) * F(n+1))，衰减但永不为零
    a, b = 1, 1
    for _ in range(step):
        a, b = b, a + b
    sign = 1 if step % 2 == 0 else -1
    return sign / (a * b + 1e-15)


class UndefinedConstant:
    """
    一个无法被精确写下的常数。

    跟踪一个漂移中的当前值，每次 apply() 推进一步。
    当 B1 死亡（=0）时注入大幅无理数扰动以复活系统。
    """

    def __init__(self, standard_value: float, name: str = ""):
        self.standard = standard_value
        self.name = name
        self.step = 0
        self._current = standard_value  # 漂移中的当前值，不再锚定standard

    def apply(self, correction: float, B1: float) -> float:
        """
        应用校正量并推进一步。

        correction: 由 flip() 计算的乘性校正 (如 1.0 ± Δ)
        B1: 当前B1值，用于判断是否需要紧急注入
        """
        self.step += 1

        # 无理数基底：黄金比例连分数，永不终止，永不为零
        irrational_base = _golden_ratio_continued_fraction(self.step)

        # B1死亡紧急注入：当B1≈0时，注入大幅无理数扰动
        if B1 < 1e-4:
            # 用绝对值保证扰动方向为正（避免小常数变负）
            emergency = abs(irrational_base) * 0.05 * max(abs(self._current), 0.01)
        else:
            # 正常模式：微小无理数残余，保证永远不精确
            emergency = abs(irrational_base) * 1e-6 * max(abs(self._current), 1e-8)

        # 应用校正 + 无理数注入
        self._current = self._current * correction + emergency

        # 保证常数不为零（除c外，c=1.0是归一化基准）
        if self.name != "c":
            self._current = max(self._current, 1e-10)

        return self._current

    @property
    def value(self) -> float:
        return self._current

    def reset(self, value: float):
        self._current = value

    def __repr__(self):
        return f"UndefinedConstant({self.name}, current≈{self._current:.10e})"


class UndefinedPhysicsManager:
    """
    管理所有常数的未定义化状态。

    核心逻辑：B1必须维持在φ(≈0.618)附近 → 回溯调整常数。
    常数不再是输入，而是B1维持自身摇晃的输出。
    """

    PHI = _PHI_INV  # 1/φ ≈ 0.6180339887... — B1的目标值

    def __init__(self, standard_values: dict):
        self.constants = {
            name: UndefinedConstant(val, name=name)
            for name, val in standard_values.items()
        }
        # 柔性参数的漂移强度更高
        self.flexible_scale = {
            "G": 0.02, "k_B": 0.015, "lambda": 0.01, "rho_crit": 0.005,
        }
        # 刚性参数只允许极微扰动
        self.rigid_scale = {
            "alpha": 0.002, "c": 0.0005, "hbar": 0.001, "m_p": 0.0005,
        }

    def flip(self, B1: float, R_scalar: float):
        """
        核心翻转：B1残差回溯重塑所有常数。

        B1 > φ: 系统在闭合（自指过强）→ 降低引力/电磁耦合，打开结构
        B1 < φ: 系统在离散（太松散）→ 增强引力/电磁耦合，凝聚结构
        B1 ≈ 0: 系统死亡 → 紧急注入大幅无理数扰动复活
        """
        delta = B1 - self.PHI  # φ = 0.618

        # === 紧急复活模式：B1死亡 ===
        if abs(B1) < 1e-4:
            # 宇宙死了。必须用强力手段拉回来。
            # G太高（引力碾碎一切）→ 大幅降低
            # lambda太低（没有膨胀扰动）→ 大幅提升
            # alpha需要扰动（不能锁在有理数上）
            self.constants["G"].apply(0.85, B1)       # G *= 0.85，大幅削减引力
            self.constants["lambda"].apply(1.15, B1)   # lambda *= 1.15，增强暗能量扰动
            self.constants["k_B"].apply(1.10, B1)      # k_B *= 1.10，增加热噪声
            self.constants["alpha"].apply(1.02, B1)    # alpha微调（刚性参数）
            self.constants["hbar"].apply(1.03, B1)     # hbar微增（量子涨落）
            self.constants["rho_crit"].apply(0.95, B1) # 降低临界密度阈值
            # c 和 m_p 不动（归一化基准）
            self.constants["c"].apply(1.0, B1)
            self.constants["m_p"].apply(1.0, B1)
            return

        # === 正常回溯模式 ===
        for name, uconst in self.constants.items():
            if name in ("c",):
                continue  # c是归一化基准，不动

            # 校正强度：偏离越大，校正越猛
            # 用 tanh 限制，防止过校正
            error = -delta  # 反向：B1低→正值→增强；B1高→负值→削弱
            strength = np.tanh(error * 2.0)  # tanh限制在(-1, 1)

            if name in ("G", "alpha", "m_p"):
                # 引力/电磁/质量：B1低时增强（凝聚），B1高时削弱（打开）
                correction = 1.0 + strength * 0.02
            elif name in ("lambda", "k_B"):
                # 暗能量/热噪声：B1低时减弱（减少扰动），B1高时增强（打破闭环）
                correction = 1.0 - strength * 0.015
            elif name == "hbar":
                # 普朗克常数：B1低时增强（更多量子涨落=更多结构可能性）
                correction = 1.0 + strength * 0.01
            elif name == "rho_crit":
                correction = 1.0 - strength * 0.008
            else:
                correction = 1.0

            uconst.apply(correction, B1)

    def read_all(self) -> dict:
        """返回当前所有常数的逼近值"""
        return {name: uconst.value for name, uconst in self.constants.items()}

    def read(self, key: str) -> float:
        return self.constants[key].value

    def items(self):
        return {k: v.value for k, v in self.constants.items()}.items()

    def to_dict(self) -> dict:
        return self.read_all()

    def is_habitable(self, B1: float) -> bool:
        """
        可居住 = B1在(0.05, 1.0)且非零。

        B1 = 0 → 死（无自指）
        B1 ∈ (0.05, 1.0) → 活着（自指在0.618附近振荡）
        B1 ≥ 1.0 → 死（完美周期性，无新信息）
        """
        return 0.05 < B1 < 1.0

# 物理常数定义 (归一化单位)
# G=0.01: 引力弱化100倍，允许结构形成而不被压碎
PHYSICAL_CONSTANTS = {
    "c": 1.0,               # 光速 (归一化)
    "G": 0.01,              # 引力常数 (归一化，弱化100倍)
    "hbar": 0.1,            # 普朗克常数 (控制量子涨落幅度)
    "k_B": 0.01,            # 玻尔兹曼常数 (控制热噪声)
    "alpha": 1/137,         # 精细结构常数 (电磁耦合强度)
    "lambda": 0.7,          # 宇宙学常数 (暗能量密度)
    "m_p": 1.0,             # 质子质量基准
    "rho_crit": 1.0         # 临界密度参数
}

# 模拟参数
SIMULATION_PARAMS = {
    "dt": 0.01,             # 时间步长
    "max_steps": 1000,      # 最大演化步数
    "embedding_dim": 128,   # 结构嵌入维度
    "memory_dim": 64,       # 记忆向量维度
    "residual_window": 100, # 残差滑动窗口长度
    "metrics_interval": 10, # 指标计算间隔
    "matter_coupling": 0.0001,  # 结构嵌入→能量密度 (随G同比弱化)
    "warmup_steps": 500,      # predictor预热步数（前N步不用predictor）
    "train_every": 50,        # 每N步在线训练一次predictor
    "gravitational_attraction": 0.0015,  # 引力吸引后处理 (随G同比弱化)
}

# LLM参数
LLM_CONFIG = {
    "base_url": "http://localhost:28000/v1",
    "embedding_url": "http://localhost:28001/v1/embeddings",
    "model": "glm4",     # 默认模型名
    "max_tokens": 1024,       # 增加token限制，避免截断
    "api_key": "tp-cp16lhqjskpx504w0gn6sipc3lpndh4bj6wfpmyq3uofg1ze",
    "temperature_base": None  # 运行时由hbar决定
}


class PhysicalConstants:
    """
    物理常数管理器

    支持三种初始化模式：
    - standard: 使用我们宇宙的标准常数
    - habitable_random: 在可居住窗口内随机采样，保证意识涌现的可能性
    - evolving: 从标准值开始，允许柔性常数随残差场漂移

    常数分为刚性（alpha, c, hbar, m_p）和柔性（G, k_B, lambda, rho_crit）。
    漂移后自动检查可居住性，不满足则回退。
    """

    # 标准值（我们宇宙的归一化常数，G弱化100倍）
    STANDARD = {
        "c": 1.0, "G": 0.01, "hbar": 0.1, "k_B": 0.01,
        "alpha": 1 / 137, "lambda": 0.7, "m_p": 1.0, "rho_crit": 1.0
    }

    # 刚性参数：只允许极小扰动 (±5%)
    RIGID_PARAMS = frozenset({"alpha", "c", "hbar", "m_p"})

    # 柔性参数：可较宽范围变化
    FLEXIBLE_PARAMS = frozenset({"G", "k_B", "lambda", "rho_crit"})

    # 可居住窗口
    HABITABLE_RANGES = {
        "alpha": (0.005, 0.05),    # 原子稳定与化学复杂性的平衡
        "lambda": (0.1, 2.0),      # 避免过快膨胀或坍缩
        # G * m_p² / (hbar * c) ∈ [1e-40, 1e-36] 由 is_habitable() 计算
    }

    def __init__(self, mode: str = "standard"):
        """
        初始化物理常数

        Args:
            mode: "standard" | "habitable_random" | "evolving" | "undefined_physics"
        """
        if mode == "standard":
            self._values = dict(self.STANDARD)
            self._undefined_manager = None
        elif mode == "habitable_random":
            self._values = self._sample_habitable()
            self._undefined_manager = None
        elif mode == "evolving":
            self._values = dict(self.STANDARD)
            self._undefined_manager = None
        elif mode == "undefined_physics":
            self._values = dict(self.STANDARD)
            self._undefined_manager = UndefinedPhysicsManager(self.STANDARD)
        else:
            raise ValueError(f"未知模式: {mode}，可选: standard, habitable_random, evolving, undefined_physics")
        self.mode = mode
        self._current_B1 = 0.0  # 当前B1值，由evolve()更新

    def __getitem__(self, key):
        if self.mode == "undefined_physics" and self._undefined_manager is not None:
            return self._undefined_manager.read(key)
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def get(self, key, default=None):
        return self._values.get(key, default)

    def items(self):
        return self._values.items()

    def to_dict(self) -> dict:
        if self.mode == "undefined_physics" and self._undefined_manager is not None:
            return self._undefined_manager.read_all()
        return dict(self._values)

    def drift(self, residual_scalar: float, dt: float = 1.0):
        """
        柔性常数随残差场微小漂移

        用残差的一维标量驱动柔性常数的随机游走 + 残差耦合。
        漂移后检查可居住性，不满足则回退到漂移前的值。

        Args:
            residual_scalar: 残差场的标量值（如残差能量的均值）
            dt: 时间步长
        """
        old = dict(self._values)

        for key in self.FLEXIBLE_PARAMS:
            # 随机游走分量
            noise = np.random.randn() * 0.001 * dt
            # 残差耦合分量
            coupling = residual_scalar * 0.0001 * dt
            self._values[key] += noise + coupling

        # 可居住性检查：不满足则回退
        if not self.is_habitable():
            self._values = old

    def _sample_habitable(self) -> dict:
        """在可居住窗口内随机采样常数"""
        v = dict(self.STANDARD)
        v["alpha"] = np.random.uniform(*self.HABITABLE_RANGES["alpha"])
        v["lambda"] = np.random.uniform(*self.HABITABLE_RANGES["lambda"])
        return v

    def undefined_read(self, B1: float, R_scalar: float) -> dict:
        """
        未定义物理模式的核心方法：B1残差回溯重塑所有常数。

        每个常数都变成逼近序列的下一项，永远不抵达精确值。

        Args:
            B1: 当前Betti-1值（目标：维持在0.618附近）
            R_scalar: 当前残差标量（预测器漏掉的那部分）

        Returns:
            当前所有常数的逼近值
        """
        if self._undefined_manager is None:
            return self._values

        self._current_B1 = B1
        self._undefined_manager.flip(B1, R_scalar)

        # 同步更新内部字典（用于prompt等场景）
        new_values = self._undefined_manager.read_all()
        self._values.update(new_values)

        return new_values

    def is_habitable(self) -> bool:
        """
        检查当前常数组合是否可能支持自指递归（意识）

        未定义物理模式下，居住性由B1决定而非常数窗口。
        """
        if self.mode == "undefined_physics" and self._undefined_manager is not None:
            return self._undefined_manager.is_habitable(self._current_B1)

        a = self._values["alpha"]
        l = self._values["lambda"]
        G = self._values["G"]
        m_p = self._values["m_p"]
        hbar = self._values["hbar"]
        c = self._values["c"]

        ratio = G * m_p ** 2 / (hbar * c)
        standard_ratio = self.STANDARD["G"] * self.STANDARD["m_p"] ** 2 / (
            self.STANDARD["hbar"] * self.STANDARD["c"])

        alpha_ok = self.HABITABLE_RANGES["alpha"][0] <= a <= self.HABITABLE_RANGES["alpha"][1]
        lambda_ok = self.HABITABLE_RANGES["lambda"][0] <= l <= self.HABITABLE_RANGES["lambda"][1]
        ratio_ok = 0.01 <= ratio / standard_ratio <= 100.0

        return alpha_ok and lambda_ok and ratio_ok

    def __repr__(self):
        habitable = "✓" if self.is_habitable() else "✗"
        if self.mode == "undefined_physics":
            return f"PhysicalConstants(mode={self.mode}, B1≈{self._current_B1:.4f}, habitable={habitable})"
        return f"PhysicalConstants(mode={self.mode}, habitable={habitable})"
