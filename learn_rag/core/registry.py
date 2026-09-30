"""
core.registry —— 一个通用注册表，负责"配置字典 -> 组件实例"。

为什么需要它：
  RAG 的每一层都有多种可选实现（3 种切分器、3 种向量模型、2 种检索器……）。
  如果在 pipeline 里写 if cfg["type"] == "bm25": ... elif ...，
  那么每加一个实现都要改 pipeline —— 这是典型的"信息泄漏"。

做法：
  用一个**通用**注册表管所有类型的组件（不是 EmbedderRegistry + RetrieverRegistry
  + ChunkerRegistry 各写一份），命名空间用字符串区分。
  接口只有 register / build 两个方法，但背后帮调用方处理了：
  依赖注入（把已构造好的组件按参数名注入）、默认值合并、清晰的报错信息。
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

class Registry:
    """
    全局组件注册表（namespace -> name -> 工厂）
    """

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, namespace: str, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        装饰器写法：@registry.register("retriever", "bm25")
        """

        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            self._factories.setdefault(namespace, {})[name] = factory
            return factory

        return deco

    def build(self, namespace: str, spec: dict[str, Any] | None, **injected: Any) -> Any:
        """
        按 spec={"type": "xxx", 其余为构造参数} 创建组件

        injected 是"上下文依赖"（如已构建好的 llm / encoder / index），
        只有当目标工厂确实声明了同名参数时才注入 —— 调用方因此可以无脑把所有可能用到的依赖都传进来，不必关心具体实现需要哪些。
        """

        if not spec:
            raise ValueError(f"[{namespace}] 配置为空")
        spec = dict(spec)
        name = spec.pop("type", None)
        if name is None:
            raise ValueError(f"[{namespace}] 配置缺少 type 字段: {spec}")

        table = self._factories.get(namespace, {})
        if name not in table:
            raise KeyError(f"[{namespace}] 未注册的实现 '{name}'，可选：{sorted(table)}")
        factory = table[name]

        import inspect

        params = inspect.signature(factory).parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = dict(spec)
        for key, value in injected.items():
            if key in params or accepts_kwargs:
                kwargs.setdefault(key, value)
        try:
            return factory(**kwargs)
        except TypeError as exc:
            raise TypeError(f"构建 [{namespace}:{name}] 失败: {exc}") from exc

    def options(self, namespace: str) -> list[str]:
        return sorted(self._factories.get(namespace, {}))

# 全系统共用一个实例；各模块 import 后用装饰器注册自己
registry = Registry()