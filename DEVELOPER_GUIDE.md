# ModularRAG — Developer Guide: Adding New Components

This guide shows you exactly how to plug a new implementation into each part of the pipeline by implementing the right interface.

---

## The Golden Rule

Every new component follows the same 3-step pattern:

```
1. Create a class that inherits from the right ABC in interfaces/
2. Implement the required abstract method(s)
3. Register it — either via config.yaml or by passing it directly
```

---

## 0. Config-First Approach

Before writing any code, check if `config.yaml` already supports what you need.
Many changes (model name, top-k values, backend switch) require **zero code changes**:

```yaml
# config.yaml — change the backend without touching any .py file
pipeline:
  llm_backend: "ollama"   # was "local"
  streaming: true

retrieval:
  vector_top_k: 10        # was 5
  fusion_top_k: 5
```

Only write a new class when you need behaviour that `config.yaml` can't express.

---

## 1. New Document Loader

**Interface:** `BaseDocumentLoader` → must implement `get_document_updates()`

**Example:** A loader that reads from an S3 bucket instead of local disk.

```python
# ingestion/s3_loader.py

from interfaces import BaseDocumentLoader

class S3DocumentLoader(BaseDocumentLoader):
    """Loads documents from an AWS S3 bucket."""

    def __init__(self, bucket_name: str) -> None:
        self._bucket_name = bucket_name

    def get_document_updates(
        self, year: int, branch: str, db_dir: str
    ) -> tuple[bool, list, list[str], dict[str, str], str]:
        # Your S3 logic here — list objects, compare etags, download changed files...
        needs_update = True
        changed_docs = [...]   # list of LlamaIndex Document objects
        deleted_files = []
        current_hashes = {"s3://my-bucket/file.pdf": "abc123"}
        state_log = f"{db_dir}/{year}_{branch}_hash_state.json"
        return needs_update, changed_docs, deleted_files, current_hashes, state_log
```

**Plug it in** — sub-class `VectorIndexBuilder` and override the loader call.

---

## 2. New Embedding Model

**Interface:** `BaseEmbedder` → must implement `get_model()`

**Example:** OpenAI text-embedding instead of local HuggingFace.

```python
# ingestion/openai_embedder.py

from interfaces import BaseEmbedder
from llama_index.embeddings.openai import OpenAIEmbedding

class OpenAIEmbedder(BaseEmbedder):
    """Uses OpenAI's API for embeddings."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    def get_model(self) -> OpenAIEmbedding:
        return OpenAIEmbedding(model=self._model)
```

**Plug it in** — pass it directly to `VectorIndexBuilder`:
```python
from ingestion.openai_embedder import OpenAIEmbedder
from ingestion.build_vector_index import VectorIndexBuilder

index = VectorIndexBuilder(embedder=OpenAIEmbedder()).build(4, "ds")
```

> ✅ `VectorIndexBuilder` already accepts an `embedder` argument — no other code changes needed.

---

## 3. New LLM

**Interface:** `BaseLLMLoader` → must implement `get_llm()`

Two loaders already ship with the project:

| Class | File | Backend |
|---|---|---|
| `QuantizedLocalLLM` | `llm_loaders/local_llm_loader.py` | HuggingFace + 4-bit bitsandbytes |
| `OllamaLoader` | `llm_loaders/ollama_loader.py` | Ollama server (streaming ✅) |

**To add a new one** (e.g. OpenAI):

```python
# llm_loaders/openai_llm_loader.py

from interfaces import BaseLLMLoader
from llama_index.llms.openai import OpenAI

class OpenAILLMLoader(BaseLLMLoader):
    """Loads an OpenAI GPT model."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1) -> None:
        self._model = model
        self._temperature = temperature

    def get_llm(self) -> OpenAI:
        return OpenAI(model=self._model, temperature=self._temperature)
```

**Plug it in** — add a branch to `_build_llm()` in `query_engine.py`:
```python
# query_engine.py — _build_llm() factory
if backend == "openai":
    return OpenAILLMLoader(model=cfg.openai_llm.model)
```

Then set `pipeline.llm_backend: "openai"` in `config.yaml` and add an `openai_llm:` section.

> ⚠️ **Important:** Do NOT call `super().__init__()` with keyword arguments.
> `BaseLLMLoader` is a plain ABC — its only parent is `object`. Store your
> arguments manually as `self._attr = value`.

---

## 4. New Retriever

**Interface:** `BaseRetrieverBuilder` → must implement `build(index)`

**Example:** A pure vector-only retriever (no BM25).

```python
# retrieval/vector_only_retriever.py

from interfaces import BaseRetrieverBuilder
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever

class VectorOnlyRetrieverBuilder(BaseRetrieverBuilder):
    """Dense vector retrieval only — no BM25."""

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def build(self, index: VectorStoreIndex) -> BaseRetriever:
        return VectorIndexRetriever(index=index, similarity_top_k=self._top_k)
```

**Plug it in** — replace the retriever step in `setup_query_engine()`:
```python
from retrieval.vector_only_retriever import VectorOnlyRetrieverBuilder

hybrid_retriever = VectorOnlyRetrieverBuilder(top_k=10).build(index)
```

---

## 5. New Index Builder

**Interface:** `BaseIndexBuilder` → must implement `build(year, branch)`

**Example:** An in-memory index (no ChromaDB) for fast local testing.

```python
# ingestion/in_memory_index_builder.py

from interfaces import BaseIndexBuilder, BaseEmbedder
from ingestion.build_vector_index import HuggingFaceEmbedder
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

class InMemoryIndexBuilder(BaseIndexBuilder):
    """Builds a transient in-memory index — useful for quick testing."""

    def __init__(self, data_dir: str, embedder: BaseEmbedder | None = None) -> None:
        self._data_dir = data_dir
        self._embedder = embedder or HuggingFaceEmbedder()

    def build(self, year: int, branch: str) -> VectorStoreIndex:
        docs = SimpleDirectoryReader(self._data_dir).load_data()
        return VectorStoreIndex.from_documents(
            docs, embed_model=self._embedder.get_model()
        )
```

**Plug it in:**
```python
from ingestion.in_memory_index_builder import InMemoryIndexBuilder

index = InMemoryIndexBuilder(data_dir="data/4th_yr_ds").build(4, "ds")
```

---

## 6. New Prompt

**Interface:** `BasePromptProvider` → must implement `get_prompt()`

`prompts/base.py` already contains two template strings — `_QA_TEMPLATE` (strict, citation-only) and `_NEW_TEMPLATE` (detailed, structured explanation). Switch between them by changing which one `CustomPromptProvider.get_prompt()` returns.

**To add a completely new style:**

```python
# prompts/concise_prompt.py

from interfaces import BasePromptProvider
from llama_index.core import PromptTemplate

_CONCISE_TEMPLATE = (
    "Answer the following question in 2-3 sentences using only the sources below.\n\n"
    "SOURCES:\n{context_str}\n\n"
    "QUESTION: {query_str}\n\nANSWER:"
)

class ConcisePromptProvider(BasePromptProvider):
    def get_prompt(self) -> PromptTemplate:
        return PromptTemplate(_CONCISE_TEMPLATE)
```

**Plug it in** — replace the prompt step in `setup_query_engine()`:
```python
from prompts.concise_prompt import ConcisePromptProvider

qa_prompt = ConcisePromptProvider().get_prompt()
```

---

## Quick Reference

| What you want | Interface to implement | Required method |
|---|---|---|
| New document source | `BaseDocumentLoader` | `get_document_updates(year, branch, db_dir)` |
| New embedding model | `BaseEmbedder` | `get_model()` |
| New LLM | `BaseLLMLoader` | `get_llm()` |
| New retrieval strategy | `BaseRetrieverBuilder` | `build(index)` |
| New index backend | `BaseIndexBuilder` | `build(year, branch)` |
| New prompt style | `BasePromptProvider` | `get_prompt()` |

---

## Checklist for any new component

- [ ] Check `config.yaml` first — you might not need to write a class at all
- [ ] Create a new `.py` file in the appropriate directory
- [ ] `from interfaces import <TheRightABC>`
- [ ] `class MyNewThing(TheRightABC):`
- [ ] Implement every `@abstractmethod` (Python will error at instantiation if you miss one)
- [ ] Store constructor args as `self._attr` — do **not** call `super().__init__()` with kwargs
- [ ] Add type hints and a docstring to your class and method(s)
- [ ] Add a test to `tests/test_interfaces.py`: `assert issubclass(MyNewThing, TheRightABC)`
- [ ] Register it — either add a branch to `_build_llm()` in `query_engine.py`, or pass it as a constructor argument
