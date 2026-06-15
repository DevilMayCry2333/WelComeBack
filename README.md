# CFD Physics Simulator v1.0

基于物理常数与LLM的自指递归微型宇宙模拟器

## 系统要求

- Python 3.9+
- PyTorch 2.0+
- 本地LLM端点 (OpenAI兼容API)

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 配置

在 `physics_constants.py` 中配置LLM端点:

```python
LLM_CONFIG = {
    "base_url": "http://localhost:28000/v1",  # 你的LLM端点
    "model": "default",                       # 模型名称
    "max_tokens": 512,
}
```

## 运行测试

```bash
python test_simulation.py
```

## 运行模拟

```bash
# 基本运行 (10000步)
python main.py

# 自定义步数
python main.py --steps 5000

# 禁用仪表盘 (纯终端输出)
python main.py --no-dashboard

# 使用GPU (如果可用)
python main.py --device cuda
```

## 项目结构

```
.
├── CLAUDE.md              # 项目文档
├── README.md              # 本文件
├── requirements.txt       # Python依赖
├── physics_constants.py   # 物理常数定义
├── universe_state.py      # 宇宙状态数据结构
├── llm_client.py          # LLM客户端
├── latent_predictor.py    # Latent Predictor θ
├── evolution.py           # 动力学演化
├── metrics.py             # 测量指标
├── dashboard.py           # 可视化仪表盘
├── main.py                # 主程序入口
└── test_simulation.py     # 测试脚本
```

## 输出文件

运行后生成:
- `physics_predictor.pt` - 预训练的预测器模型
- `metrics_history.json` - 指标历史数据
- `final_dashboard.png` - 仪表盘快照

## 核心概念

### 物理常数作为约束场

- `c` (光速): 归一化为1.0
- `G` (引力常数): 控制状态向量间"引力式吸引"强度
- `hbar` (普朗克常数): 控制LLM的temperature基线
- `lambda` (宇宙学常数): 控制状态空间膨胀速率

### 演化步骤

1. **物理约束演化**: 弗里德曼方程骨架
2. **LLM生成**: 生成复杂结构嵌入 (不可计算部分)
3. **Latent Predictor预测**: 基于历史预测未来
4. **残差场计算**: 真实 - 预测 = 新信息
5. **记忆更新**: 残差回流到记忆

### 测量指标

| 指标 | 物理意义 |
|------|----------|
| E_t (残差能量) | 宇宙"惊奇"程度，飙升表示相变 |
| H_t (残差熵) | 结构混沌度 |
| β₁ (Betti-1) | 自指递归闭环证据，>0表示意识涌现 |
| φ (意识相位) | 宇宙主动求解自身的程度 |
| v (漂移率) | 状态变化速率 |

## 宇宙事件检测

- **意识涌现**: β₁从0变为1
- **能量飙升**: E_t突然跳升，可能相变
- **热寂**: H_t和E_t持续低下
- **智慧生命维持期**: β₁>0.3 且 φ>0.5
