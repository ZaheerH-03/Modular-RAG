# ModularRAG 🎓

A fully modular, interface-driven **Retrieval-Augmented Generation (RAG)** system built for college academic content. Ask questions about your course notes and get grounded, cited answers — powered by a local quantized LLM or Ollama, no OpenAI needed.

---

## How It Works

```
PDF Notes ──► Incremental Ingestion ──► ChromaDB (persistent)
                                              │
              User Query ────────────► Hybrid Retriever (Vector + BM25 cached)
                                              │
                                       Response Synthesizer
                                       (Local LLM or Ollama + Prompt)
                                              │
                                       ✅ Grounded Answer + Sources
```

1. **Ingestion** — PDFs are hashed (MD5). Only new or modified files are re-processed on each run.
2. **Semantic Chunking** — `SemanticSplitterNodeParser` splits on topic boundaries, not arbitrary character limits.
3. **Hybrid Retrieval** — Combines dense vector search (BGE-M3) and BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF). BM25 nodes are cached to disk so reconstruction from ChromaDB only happens once.
4. **Pluggable LLM** — Switch between a 4-bit quantized local Llama model (`bitsandbytes`) or any model served by a locally running Ollama instance — all via `config.yaml`.
5. **Configurable Prompt** — Multiple prompt styles available in `prompts/base.py`, switchable without code changes.

---

## Project Structure

```
ModularRAG/
│
├── interfaces/                  # Abstract Base Classes — the contracts
│   ├── base_loader.py           #   BaseDocumentLoader
│   ├── base_embedder.py         #   BaseEmbedder
│   ├── base_llm.py              #   BaseLLMLoader
│   ├── base_retriever.py        #   BaseRetrieverBuilder
│   ├── base_index.py            #   BaseIndexBuilder
│   └── base_prompt.py           #   BasePromptProvider
│
├── ingestion/
│   ├── data_loader.py           # Incremental MD5 change detection
│   └── build_vector_index.py    # Embedding + ChromaDB upsert + BM25 cache invalidation
│
├── retrieval/
│   └── hybrid_retriever.py      # Hybrid RRF (Vector + BM25) with disk-cached nodes
│
├── llm_loaders/
│   ├── local_llm_loader.py      # 4-bit quantized HuggingFace LLM (bitsandbytes)
│   └── ollama_loader.py         # Ollama server-backed LLM (streaming supported)
│
├── prompts/
│   └── base.py                  # QA prompt templates (swap via CustomPromptProvider)
│
├── tests/
│   └── test_interfaces.py       # Structural tests — no GPU required
│
├── data/                        # PDFs organized by year/branch  [git-ignored]
│   ├── 1st_yr_ds/
│   └── 4th_yr_ds/
├── chromadb/                    # Persistent vector store + BM25 cache  [git-ignored]
├── models/                      # Local model weights              [git-ignored]
├── config.yaml                  # ⚙️  All settings — edit this, not the code
├── config.py                    # Typed config loader (reads config.yaml)
├── query_engine.py              # 🚀 Main entry point
├── DEVELOPER_GUIDE.md           # How to add new components
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Add your PDFs

Place your notes under `data/` following the `{nth}_yr_{branch}` naming convention:

```
data/
├── 1st_yr_ds/    ← 1st year Data Science notes
├── 2nd_yr_ds/
└── 4th_yr_ds/
```

### 3. Configure

Open **`config.yaml`** and set your preferences — this is the **only file you need to edit**:

```yaml
paths:
  base_dir: "E:/your/project/path"     # ← change this to your actual path
  embedding_model: "models/bge-m3"

local_llm:
  model: "meta-llama/Llama-3.2-3B-Instruct"

pipeline:
  year: 4
  branch: "ds"
  llm_backend: "local"    # "local" = HuggingFace | "ollama" = Ollama server
  streaming: false         # true only works with ollama backend
```

### 4. Run

```bash
python query_engine.py
```

The first run ingests and indexes your documents. Subsequent runs skip unchanged files and load the BM25 node cache instantly.

---

## Switching to Ollama

Start the Ollama server, pull a model, then update `config.yaml`:

```bash
ollama pull qwen2.5:7b
```

```yaml
# config.yaml
ollama_llm:
  model: "qwen2.5:7b"

pipeline:
  llm_backend: "ollama"
  streaming: true
```

No code changes needed.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | [LlamaIndex](https://www.llamaindex.ai/) 0.10+ |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (persistent) |
| Embedding Model | [BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3) (CPU, local) |
| LLM (local) | HuggingFace Transformers + 4-bit NF4 (`bitsandbytes`) |
| LLM (server) | [Ollama](https://ollama.com/) — any supported model |
| Sparse Retrieval | BM25 (`llama-index-retrievers-bm25`, disk-cached) |
| Language | Python 3.10+ |
| Acceleration | PyTorch + CUDA |

---

## Silo Architecture

Data is stored in **separate ChromaDB collections** per year and branch (`{year}_yr_{branch}`). Access is **cumulative** — querying year 4 automatically includes years 1–3 in the index.

```
query year=4, branch="ds"
    └── indexes: 1st_yr_ds + 2nd_yr_ds + 3rd_yr_ds + 4th_yr_ds
```

---

## Running Tests

The test suite verifies interface compliance without requiring a GPU or any model downloads:

```bash
python -m pytest tests/test_interfaces.py -v
```

---

## Adding New Components

The entire pipeline is built on ABCs. To swap in a new LLM, embedder, retriever, or any other component, see **[DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)**.