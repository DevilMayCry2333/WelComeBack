"""
Latent Predictor θ — 基于已知物理历史预测未来结构嵌入
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PhysicsPredictor(nn.Module):
    """
    物理预测器神经网络

    编码宇宙的因果模式，基于记忆序列和意识深度预测结构嵌入
    """

    def __init__(self, memory_dim: int = 64, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()

        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # RNN处理记忆序列
        self.rnn = nn.GRU(
            input_size=memory_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 全连接层，结合RNN输出和意识深度
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出归一化到[-1,1]
        )

        # 残差连接的投影层
        self.residual_proj = nn.Linear(output_dim, output_dim)

    def forward(self, memory_sequence: torch.Tensor,
                consciousness_depth: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            memory_sequence: (batch, seq_len, memory_dim) 记忆序列
            consciousness_depth: (batch, 1) 意识深度

        Returns:
            predicted_embedding: (batch, output_dim) 预测的结构嵌入
        """
        # RNN处理记忆序列
        rnn_out, h_n = self.rnn(memory_sequence)

        # 取最后一个时间步的隐藏状态
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # 拼接意识深度
        combined = torch.cat([last_hidden, consciousness_depth], dim=1)

        # 预测结构嵌入
        predicted = self.fc(combined)

        return predicted

    def predict_with_uncertainty(self, memory_sequence: torch.Tensor,
                                 consciousness_depth: torch.Tensor,
                                 n_samples: int = 10) -> tuple:
        """
        带不确定性的预测

        使用Monte Carlo Dropout估计预测不确定性
        """
        self.train()  # 启用dropout

        predictions = []
        for _ in range(n_samples):
            pred = self.forward(memory_sequence, consciousness_depth)
            predictions.append(pred)

        predictions = torch.stack(predictions)  # (n_samples, batch, output_dim)

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        self.eval()  # 关闭dropout

        return mean_pred, uncertainty


class PredictorTrainer:
    """
    预测器训练器
    """

    def __init__(self, predictor: PhysicsPredictor, learning_rate: float = 1e-4):
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, memory_batch: torch.Tensor,
                   consciousness_batch: torch.Tensor,
                   target_embedding: torch.Tensor) -> float:
        """
        单步训练

        Args:
            memory_batch: (batch, seq_len, memory_dim)
            consciousness_batch: (batch, 1)
            target_embedding: (batch, output_dim) 目标嵌入

        Returns:
            loss_value: 损失值
        """
        self.predictor.train()

        # 前向传播
        predicted = self.predictor(memory_batch, consciousness_batch)

        # 计算损失
        loss = self.criterion(predicted, target_embedding)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
