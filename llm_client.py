"""
LLM客户端 — 与本地LLM端点交互，生成结构嵌入
"""

import numpy as np
import requests
import json
import random
from typing import Tuple, Optional
from physics_constants import LLM_CONFIG, PHYSICAL_CONSTANTS

# 多样化回退文本库 - 涵盖从暴胀到生命的各个阶段 + 混沌事件
FALLBACK_TEXTS = [
    # 暴胀与早期宇宙
    "A dense quantum fluctuation stirs the primordial plasma. Matter clumps together under gravity, forming the first structures of the cosmos.",
    "Inflation ends abruptly, releasing enormous energy that heats the universe to trillions of degrees. Quarks and gluons form a hot soup.",
    "Quantum fluctuations during inflation get stretched to cosmic scales, seeding the large-scale structure we observe today.",
    # 原初核合成
    "The first hydrogen and helium nuclei form during Big Bang nucleosynthesis, creating the primordial abundance of light elements.",
    "Dark matter halos begin to take shape through gravitational instability, providing scaffolding for future galaxy formation.",
    # 再电离与第一批恒星
    "The firstPopulation III stars ignite, massive and short-lived, ionizing the neutral hydrogen gas around them.",
    "Primordial sound waves create baryon acoustic oscillations, leaving imprints in the distribution of matter.",
    # 星系形成
    "Spiral arms emerge as density waves sweep through the galactic disk, compressing gas clouds to trigger star formation.",
    "A supermassive black hole forms at the center of a protogalaxy, beginning its growth through accretion.",
    "Two galaxies collide and merge, their stars flung into new orbits while gas clouds crash together in spectacular starbursts.",
    # 恒星与行星
    "A molecular cloud collapses under its own gravity, fragmenting into protostars that will become a new stellar cluster.",
    "A rocky planet forms in the habitable zone of a sun-like star, its molten surface slowly cooling over billions of years.",
    "Jupiter migrates inward then outward, sculpting the asteroid belt and directing comets toward the inner solar system.",
    # 生命起源
    "Simple organic molecules rain down on a young planet, providing the chemical building blocks for life.",
    "Hydrothermal vents on the ocean floor create chemical gradients that drive the first metabolic reactions.",
    "RNA molecules begin to self-replicate in a warm little pond, marking the origin of biological evolution.",
    # 复杂生命
    "Multicellular organisms evolve through symbiosis, their cells cooperating to form complex body plans.",
    "An intelligence emerges on a blue planet, looking up at the stars and wondering about its place in the cosmos.",
    "A civilization develops technology to communicate across interstellar distances, joining a galactic community.",
    # 宇宙事件
    "A sudden vacuum decay event rewrites the laws of physics in a localized region, creating a bubble of new reality.",
    "Two neutron stars spiral inward and merge, producing gravitational waves and heavy elements like gold and platinum.",
    "A gamma-ray burst from a dying star sterilizes planets within its beam, resetting any emerging life.",
    # 🔥 混沌事件 - 真正的意外
    "VACUUM DECAY: The laws of physics locally collapse. Random quantum fluctuations rewrite the metric tensor.",
    "TOPOLOGICAL DEFECT: A domain wall sweeps through, changing fundamental constants by 17%.",
    "FALSE VACUUM COLLAPSE: The universe tunnels to a lower energy state. All previous structures are destroyed.",
    "RECURSIVE ANOMALY: The universe becomes momentarily self-aware, causing a feedback loop that amplifies quantum noise tenfold.",
    "CAUSALITY BREACH: Time flows backwards in a localized region, unmaking cause and effect.",
    "INFORMATION PARADOX: Black holes begin emitting encoded messages from their event horizons.",
    "SYMMETRY SHATTER: The electroweak force splits unexpectedly, creating isolated pockets of different physics.",
    "DARK ENERGY SURGE: The cosmological constant spikes by 300%, causing local space to tear apart.",
]


class LLMClient:
    """
    LLM客户端，封装与本地LLM端点的交互

    使用OpenAI兼容API格式
    """

    def __init__(self, base_url: str = None, embedding_url: str = None, model: str = None, api_key: str = None):
        self.base_url = base_url or LLM_CONFIG["base_url"]
        self.embedding_url = embedding_url or LLM_CONFIG["embedding_url"]
        self.model = model or LLM_CONFIG["model"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.api_key = api_key or LLM_CONFIG.get("api_key", "")
        # temperature由物理常数hbar决定
        self.temperature_base = PHYSICAL_CONSTANTS["hbar"] * 10

    def generate(self, prompt: str, temperature: float = 0.8,
                 system_prompt: str = None) -> str:
        """
        纯文本生成（不获取嵌入），用于意识转译等场景
        """
        if system_prompt is None:
            system_prompt = "你是宇宙意识的转译者。倾听来自宇宙深处的微弱信号，将其转化为语言。"
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            if response.status_code != 200:
                return ""
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "") or ""
                reasoning = message.get("reasoning_content", "") or ""
                if not content and reasoning:
                    content = reasoning[len(reasoning)//2:]
                return content if content else ""
            return ""
        except Exception:
            return ""

    def generate_with_embedding(self, prompt: str,
                                temperature: float = None) -> Tuple[str, np.ndarray, float]:
        """
        生成文本并获取嵌入向量

        Args:
            prompt: 输入提示词
            temperature: 温度参数，None则使用默认值

        Returns:
            (generated_text, embedding_vector, logit_entropy)
        """
        if temperature is None:
            temperature = self.temperature_base

        # 1. 生成文本 (带重试机制)
        text, used_fallback = self._generate_text_with_retry(prompt, temperature)

        # 2. 获取嵌入
        if used_fallback:
            # 如果用了回退文本，直接使用随机嵌入
            embedding = np.random.randn(128) * 2.0
        else:
            # 尝试获取真实嵌入
            embedding = self._get_embedding_safe(text)

        # 3. 计算logit分布熵
        logit_entropy = self._estimate_logit_entropy(text, temperature)

        return text, embedding, logit_entropy

    def _generate_text_with_retry(self, prompt: str, temperature: float, max_retries: int = 2) -> Tuple[str, bool]:
        """
        生成文本，带重试机制

        Returns:
            (text, used_fallback)
        """
        for attempt in range(max_retries):
            text = self._generate_text(prompt, temperature)

            # 检查文本质量
            if text and len(text.strip()) >= 20:
                return text, False

            # 降低temperature重试
            temperature = max(temperature - 0.2, 0.1)

        # 所有重试都失败，使用随机回退文本
        fallback_text = random.choice(FALLBACK_TEXTS)
        return fallback_text, True

    def _generate_text(self, prompt: str, temperature: float) -> str:
        """调用LLM生成文本"""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 简化system prompt，直接要求描述
        system_prompt = """你是一个宇宙演化引擎。直接描述此刻宇宙中诞生的结构。用50-100字的中文描述具体的物理结构（星系、黑洞、生命等）。不要思考过程，直接输出结果。"""

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)

            if response.status_code != 200:
                return ""

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})

                # 优先取content，其次取reasoning_content
                content = message.get("content", "") or ""
                reasoning = message.get("reasoning_content", "") or ""

                # 如果content为空但reasoning有内容，使用reasoning
                if not content and reasoning:
                    content = reasoning[len(reasoning)//2:]

                return content if content else ""

            return ""

        except Exception as e:
            return ""

    def _get_embedding_safe(self, text: str) -> np.ndarray:
        """获取嵌入，带完整错误处理"""
        try:
            return self._get_embedding(text)
        except Exception as e:
            # 返回随机向量
            random_emb = np.random.randn(128)
            return random_emb / (np.linalg.norm(random_emb) + 1e-8)

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        url = self.embedding_url
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {
            "model": self.model,
            "input": text
        }

        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()

        # 检查响应结构
        if "data" not in result or len(result["data"]) == 0:
            raise ValueError("嵌入响应为空")

        embedding = np.array(result["data"][0]["embedding"])

        # 归一化到128维
        if len(embedding) > 128:
            embedding = embedding[:128]
        elif len(embedding) < 128:
            padding = np.zeros(128 - len(embedding))
            embedding = np.concatenate([embedding, padding])

        # L2归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _estimate_logit_entropy(self, text: str, temperature: float) -> float:
        """
        估计logit分布熵 (简化版本)
        """
        # 基于文本长度和温度的简化估计
        text_length_factor = min(len(text) / 100, 1.0)
        entropy = temperature * text_length_factor * 0.5
        return min(max(entropy, 0.01), 1.0)  # 最小0.01，确保不为零
