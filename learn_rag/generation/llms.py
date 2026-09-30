"""
generation.llms —— LLM 客户端。

只用标准库 urllib 实现 OpenAI 兼容协议：DeepSeek、Qwen、vLLM、Ollama、
本地部署的任何 OpenAI-compatible 服务都能直接用，换模型只改配置。

另外提供 EchoLLM：不联网、可预测，用于单元测试和"没配 API Key 也能跑通"的场景。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..core.interfaces import LLM
from ..core.registry import registry


@registry.register("llm", "openai_compat")
class OpenAICompatLLM(LLM):
    """
    任何 OpenAI 兼容的 /v1/chat/completions 服务。

    内置指数退避重试：网络抖动和限流是评测跑批时的常态，
    把重试藏在这一层，上层几百处调用都不用写 try/except —— 深类的价值
    """

    def __init__(
            self,
            model: str = "deepseek-chat",
            base_url: str | None = None,
            api_key: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            timeout: int = 120,
            max_retries: int = 3
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict[str, str]], **options: Any) -> str:
        payload = {
            "model": options.get("model", self.model),
            "messages": messages,
            "temperature": options.get("temperature", self.temperature),
            "max_tokens": options.get("max_tokens", self.max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                return body["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM 调用失败（重试 {self.max_retries} 次）：{last_error}")


@registry.register("llm", "echo")
class EchoLLM(LLM):
    """
    离线桩：原样回显最后一条用户消息的末尾片段。

    用途：单元测试、确定性回归、演示"没有真模型时链路依然完整"。
    绝不要用它跑正式评测（它不会产生有意义的答案）。
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def chat(self, messages: list[dict[str, str]], **options: Any) -> str:
        last = messages[-1]["content"] if messages else ""
        return self.prefix + last[-200:]
