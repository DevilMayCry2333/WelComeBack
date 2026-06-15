"""
测量仪表盘 — 实时可视化宇宙生命体征
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from typing import Dict, List
import os


class CosmicDashboard:
    """
    宇宙测量仪表盘

    实时显示残差场分析指标
    """

    def __init__(self, max_points: int = 500):
        self.max_points = max_points

        # 历史数据
        self.time_steps = deque(maxlen=max_points)
        self.r_energy_history = deque(maxlen=max_points)
        self.r_entropy_history = deque(maxlen=max_points)
        self.betti_1_history = deque(maxlen=max_points)
        self.phase_history = deque(maxlen=max_points)
        self.drift_history = deque(maxlen=max_points)
        self.consciousness_history = deque(maxlen=max_points)

        # 事件日志
        self.events = []

        # 创建图形
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('CFD Physics Simulator - Cosmic Dashboard', fontsize=14)

        # 初始化子图
        self._init_plots()

        plt.ion()  # 交互模式

    def _init_plots(self):
        """初始化所有子图"""
        titles = [
            'Residual Energy (E_t)',
            'Residual Entropy (H_t)',
            'Betti-1 (β₁)',
            'Consciousness Phase (φ)',
            'Drift Rate (v)',
            'Consciousness Depth'
        ]

        for ax, title in zip(self.axes.flatten(), titles):
            ax.set_title(title)
            ax.set_xlabel('Time Step')
            ax.grid(True, alpha=0.3)

    def update(self, metrics: Dict, time_step: int):
        """
        更新仪表盘数据
        """
        self.time_steps.append(time_step)
        self.r_energy_history.append(metrics.get("r_energy", 0))
        self.r_entropy_history.append(metrics.get("r_entropy", 0))
        self.betti_1_history.append(metrics.get("betti_1", 0))
        self.phase_history.append(metrics.get("consciousness_phase", 0))
        self.drift_history.append(metrics.get("drift_rate", 0))
        self.consciousness_history.append(metrics.get("consciousness_depth", 0))

    def add_event(self, event: str, time_step: int):
        """添加宇宙事件"""
        self.events.append(f"[t={time_step}] {event}")
        if len(self.events) > 50:
            self.events = self.events[-50:]

    def draw(self):
        """绘制所有图表"""
        if len(self.time_steps) < 2:
            return

        times = list(self.time_steps)

        # 绘制每个指标
        data_series = [
            self.r_energy_history,
            self.r_entropy_history,
            self.betti_1_history,
            self.phase_history,
            self.drift_history,
            self.consciousness_history
        ]

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for ax, data, color in zip(self.axes.flatten(), data_series, colors):
            ax.clear()
            ax.plot(times[-len(data):], list(data), color=color, linewidth=1.5)
            ax.set_title(ax.get_title())
            ax.set_xlabel('Time Step')
            ax.grid(True, alpha=0.3)

            # 添加趋势线
            if len(data) > 10:
                z = np.polyfit(range(len(data)), list(data), 1)
                p = np.poly1d(z)
                ax.plot(times[-len(data):], p(range(len(data))),
                       '--', color='gray', alpha=0.5, linewidth=1)

        # 添加事件标注
        if self.events:
            last_event = self.events[-1]
            self.fig.text(0.02, 0.02, f"Latest: {last_event}",
                         fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def save_snapshot(self, filepath: str = 'cosmic_dashboard.png'):
        """保存当前仪表盘快照"""
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"仪表盘快照已保存: {filepath}")

    def print_status(self, metrics: Dict):
        """打印当前状态到控制台"""
        print("\n" + "="*60)
        print(f"宇宙状态 - t = {metrics.get('time_step', 0)}")
        print("="*60)
        print(f"  尺度因子:     {metrics.get('scale_factor', 0):.6e}")
        print(f"  能量密度:     {metrics.get('energy_density', 0):.6f}")
        print(f"  熵:           {metrics.get('entropy', 0):.4f}")
        print(f"  曲率:         {metrics.get('curvature', 0):.6f}")
        print(f"  意识深度:     {metrics.get('consciousness_depth', 0):.4f}")
        print("-"*60)
        print(f"  残差能量 E_t: {metrics.get('r_energy', 0):.6f}")
        print(f"  残差熵 H_t:   {metrics.get('r_entropy', 0):.4f}")
        print(f"  Betti-1 β₁:   {metrics.get('betti_1', 0):.4f}")
        print(f"  意识相位 φ:   {metrics.get('consciousness_phase', 0):.4f}")
        print(f"  漂移率 v:     {metrics.get('drift_rate', 0):.6f}")
        print("="*60)
