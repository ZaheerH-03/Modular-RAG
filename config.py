"""
config.py
=========
Loads ``config.yaml`` from the project root and exposes typed, dataclass-based
configuration objects for every section of the pipeline.

Usage (anywhere in the project):

    from config import cfg

    print(cfg.paths.base_dir)
    print(cfg.local_llm.model)
    print(cfg.retrieval.vector_top_k)
"""

import os
from dataclasses import dataclass

import yaml

# ---------------------------------------------------------------------------
# Locate and load config.yaml
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(_CONFIG_PATH, "r") as _f:
    _raw: dict = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Typed config sections (dataclasses give attribute access + type clarity)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathsConfig:
    base_dir: str
    db_dir: str
    embedding_model: str


@dataclass(frozen=True)
class RetrievalConfig:
    vector_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    num_queries: int


@dataclass(frozen=True)
class LocalLLMConfig:
    model: str
    max_new_tokens: int
    context_window: int
    temperature: float


@dataclass(frozen=True)
class OllamaLLMConfig:
    model: str
    request_timeout: float
    context_window: int
    temperature: float


@dataclass(frozen=True)
class PipelineConfig:
    year: int
    branch: str
    llm_backend: str   # "local" | "ollama"
    streaming: bool
    test_query: str


@dataclass(frozen=True)
class AppConfig:
    """Root config object — import ``cfg`` to access all sections."""
    paths: PathsConfig
    retrieval: RetrievalConfig
    local_llm: LocalLLMConfig
    ollama_llm: OllamaLLMConfig
    pipeline: PipelineConfig


# ---------------------------------------------------------------------------
# Helpers to resolve relative paths against base_dir
# ---------------------------------------------------------------------------

def _resolve(base: str, path: str) -> str:
    """Return ``path`` as-is if absolute, otherwise join with ``base``."""
    return path if os.path.isabs(path) else os.path.join(base, path)


# ---------------------------------------------------------------------------
# Build the singleton config object
# ---------------------------------------------------------------------------

def _load() -> AppConfig:
    p = _raw["paths"]
    base = p["base_dir"]

    paths = PathsConfig(
        base_dir=base,
        db_dir=_resolve(base, p["db_dir"]),
        embedding_model=_resolve(base, p["embedding_model"]),
    )

    retrieval = RetrievalConfig(**_raw["retrieval"])
    local_llm = LocalLLMConfig(**_raw["local_llm"])
    ollama_llm = OllamaLLMConfig(**_raw["ollama_llm"])
    pipeline = PipelineConfig(**_raw["pipeline"])

    return AppConfig(
        paths=paths,
        retrieval=retrieval,
        local_llm=local_llm,
        ollama_llm=ollama_llm,
        pipeline=pipeline,
    )


cfg: AppConfig = _load()
