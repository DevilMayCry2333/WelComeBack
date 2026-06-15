"""
物理常数定义 — 基于我们宇宙的基本常数，归一化单位
"""

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

# 模拟参数
SIMULATION_PARAMS = {
    "dt": 0.01,             # 时间步长
    "max_steps": 200,       # 最大演化步数
    "embedding_dim": 128,   # 结构嵌入维度
    "memory_dim": 64,       # 记忆向量维度
    "residual_window": 100, # 残差滑动窗口长度
    "metrics_interval": 10  # 指标计算间隔
}

# LLM参数
LLM_CONFIG = {
    "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
    "embedding_url": "http://localhost:28001/v1/embeddings",
    "model": "mimo-v2.5",     # 默认模型名
    "max_tokens": 1024,       # 增加token限制，避免截断
    "api_key": "",
    "temperature_base": None  # 运行时由hbar决定
}
