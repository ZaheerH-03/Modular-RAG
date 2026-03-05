# ModularRAG — Developer Guide: Adding New Components

This guide shows you exactly how to plug a new implementation into each part of the pipeline by implementing the right interface.

---

## The Golden Rule

Every new component follows the same 3-step pattern:

```
1. Create a class that inherits from the right ABC in interfaces/
2. Implement the required abstract method(s)
3. (Optional) Add a convenience function so old call sites still work
```

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

**Plug it in** — `build_vector_index.py`:
```python
# Replace the default loader inside VectorIndexBuilder
from ingestion.s3_loader import S3DocumentLoader

builder = VectorIndexBuilder()
# Patch the loader call, or sub-class VectorIndexBuilder and override _get_updates()
```

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

**Plug it in** — pass it to `VectorIndexBuilder`:
```python
from ingestion.openai_embedder import OpenAIEmbedder
from ingestion.build_vector_index import VectorIndexBuilder

index = VectorIndexBuilder(embedder=OpenAIEmbedder()).build(4, "ds")
```

> ✅ `VectorIndexBuilder` already accepts an `embedder` argument — no other code changes needed.

---

## 3. New LLM

**Interface:** `BaseLLMLoader` → must implement `get_llm()`

**Example:** OpenAI GPT-4o instead of a local quantized model.

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

**Plug it in** — `query_engine.py`:
```python
from llm_loaders.openai_llm_loader import OpenAILLMLoader

loader = OpenAILLMLoader()
llm = loader.get_llm()
Settings.llm = llm
```

---

## 4. New Retriever

**Interface:** `BaseRetrieverBuilder` → must implement `build(index)`

**Example:** A pure vector-only retriever (no BM25), with a higher top-k.

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

**Plug it in** — `query_engine.py`:
```python
from retrieval.vector_only_retriever import VectorOnlyRetrieverBuilder

retriever = VectorOnlyRetrieverBuilder(top_k=10).build(index)
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

**Example:** A more conversational prompt that allows the LLM to elaborate.

```python
# prompts/conversational_prompt.py

from interfaces import BasePromptProvider
from llama_index.core import PromptTemplate

_CONVERSATIONAL_TEMPLATE = (
    "You are a helpful teaching assistant. Use the sources below to answer "
    "the student's question in a friendly, detailed way.\n\n"
    "SOURCES:\n{context_str}\n\n"
    "QUESTION: {query_str}\n\n"
    "ANSWER:"
)

class ConversationalPromptProvider(BasePromptProvider):
    """A friendlier, less strict prompt for general tutoring."""

    def get_prompt(self) -> PromptTemplate:
        return PromptTemplate(_CONVERSATIONAL_TEMPLATE)
```

**Plug it in** — `query_engine.py`:
```python
from prompts.conversational_prompt import ConversationalPromptProvider

qa_prompt = ConversationalPromptProvider().get_prompt()
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

- [ ] Create a new `.py` file in the appropriate directory
- [ ] `from interfaces import <TheRightABC>`
- [ ] `class MyNewThing(TheRightABC):`
- [ ] Implement every `@abstractmethod` (Python will tell you if you missed one)
- [ ] Add type hints and a docstring to your class and method(s)
- [ ] Add a test to `tests/test_interfaces.py`:
      `assert issubclass(MyNewThing, TheRightABC)`
- [ ] Swap it in at the call site in `query_engine.py` (or pass it as an argument)
