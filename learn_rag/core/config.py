"""
core.config —— 配置加载。

原则：配置是"数据"，不是"代码"。整条 RAG 链路的形态（用哪个切分器、
几路召回、是否重排、用哪个模型）全部由一份配置描述，
代码里不出现任何硬编码分支。这样做实验只需改 yaml，不需要改代码。

支持 .yaml（需 pyyaml）与 .json（零依赖）。深度合并让你可以只写差异项：
    cfg = load_config("configs/default.yaml", override="configs/exp_hybrid.yaml")
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "chunker": {"type": "recursive", "chunk_size": 400, "chunk_overlap": 80},
    "encoder": {"type": "hashing", "dimension": 512},
    "index": {"type": "flat"},
    "retriever": {
        "type": "hybrid",
        "weights": [1.0, 1.0],
        "channels": [{"type": "vector"}, {"type": "bm25"}],
    },
    "reranker": {"type": "lexical"},
    "llm": {"type": "echo"},
    "generator": {"type": "extractive"},   # 默认离线：不依赖 LLM
    "pipeline": {"top_k": 5, "candidate_k": 20},
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """深度合并。

    有一条特殊规则：如果双方都是"组件规格"（含 type 字段）且 type 不同，
    则**整体替换**而不是逐字段合并。否则把 hybrid 覆盖成 vector 时，
    hybrid 的 channels/weights 会残留下来传给 VectorRetriever 导致构造失败。
    这条规则让实验配置可以只写差异项。
    """
    out = dict(base)
    for key, value in patch.items():
        old = out.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            if "type" in value and "type" in old and value["type"] != old["type"]:
                out[key] = value
            else:
                out[key] = _deep_merge(old, value)
        else:
            out[key] = value
    return out


def _read_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("读取 yaml 配置需要 pyyaml：pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


_ENV_LOADED = False


def load_env(path: str | Path | None = None, *, verbose: bool = False) -> str | None:
    """加载 .env 文件到环境变量（幂等：多次调用只生效一次）。

    查找顺序：
      1. 显式传入的 path
      2. 当前工作目录下的 .env
      3. 项目根目录（learn_rag 包的上一级）下的 .env
    找到第一个就停。

    **优先级：已存在的环境变量 > .env 文件**（override=False）。
    这样 shell / PyCharm 里临时 export 的值能覆盖 .env，
    换模型做对照实验时不用改文件。反过来的话，.env 会把你临时设的值悄悄盖掉。

    python-dotenv 是可选依赖：没装时静默跳过，不影响用 shell 环境变量的人。
    返回实际加载的文件路径，没加载则返回 None。
    """
    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        if verbose:
            logger.info("未安装 python-dotenv，跳过 .env 加载（pip install python-dotenv）")
        _ENV_LOADED = True
        return None

    candidates = [Path(path)] if path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            _ENV_LOADED = True
            if verbose:
                logger.info("已加载 %s", candidate)
            return str(candidate)
    _ENV_LOADED = True
    if verbose:
        logger.info("未找到 .env 文件（可选），使用当前环境变量")
    return None


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归展开 ${VAR} / ${VAR:-默认值}。

    让 yaml 可以直接引用环境变量，于是 debug 文件和 CLI 共用**同一组**
    环境变量（LLM_API_KEY、EMBEDDING_MODEL……），不用在 yaml 里再抄一遍，
    也避免把密钥写进会被提交的配置文件。
    展开后是纯数字的字符串会转成 int/float（比如 dimension: ${EMBEDDING_DIM}）。
    """
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str) and "${" in value:
        expanded = _ENV.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
        if re.fullmatch(r"-?\d+", expanded):
            return int(expanded)
        if re.fullmatch(r"-?\d+\.\d+", expanded):
            return float(expanded)
        return expanded
    return value


def load_config(*paths: str | Path, **overrides: Any) -> dict[str, Any]:
    """按顺序加载并深度合并多份配置，最后再叠加关键字覆盖。"""
    # 必须在展开 ${VAR} 之前加载 .env，否则 yaml 里引用的变量全是空的
    load_env()
    cfg = dict(DEFAULT_CONFIG)
    for path in paths:
        if path and os.path.exists(path):
            cfg = _deep_merge(cfg, _expand_env(_read_file(path)))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
