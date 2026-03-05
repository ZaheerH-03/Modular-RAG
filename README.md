# ModularRAG 🎓

A fully modular, interface-driven **Retrieval-Augmented Generation (RAG)** system built for college academic content. Ask questions about your course notes and get grounded, cited answers — powered by a local quantized LLM, no OpenAI needed.

---

## How It Works

```
PDF Notes ──► Incremental Ingestion ──► ChromaDB (persistent)
                                              │
              User Query ────────────► Hybrid Retriever (Vector + BM25)
                                              │
                                       Response Synthesizer
                                       (Local LLM + Strict Prompt)
                                              │
                                       ✅ Grounded Answer + Sources
```

1. **Ingestion** — PDFs are hashed (MD5). Only new or modified files are re-processed on each run.
2. **Semantic Chunking** — `SemanticSplitterNodeParser` splits on topic boundaries, not arbitrary character limits.
3. **Hybrid Retrieval** — Combines dense vector search (BGE-M3) and BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF).
4. **Local LLM** — A 4-bit quantized Llama model runs entirely on-device (GPU via `bitsandbytes`).
5. **Strict Prompt** — The LLM is instructed to answer only from retrieved sources, with inline citations.

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
│   └── build_vector_index.py    # Embedding + ChromaDB upsert
│
├── retrieval/
│   └── hybrid_retriever.py      # Hybrid RRF (Vector + BM25)
│
├── llm_loaders/
│   └── local_llm_loader.py      # 4-bit quantized local LLM wrapper
│
├── prompts/
│   └── base.py                  # Strict source-grounded QA prompt
│
├── tests/
│   └── test_interfaces.py       # Structural tests (no GPU required)
│
├── data/                        # PDFs organized by year/branch  [git-ignored]
│   ├── 1st_yr_ds/
│   └── 4th_yr_ds/
├── chromadb/                    # Persistent vector store          [git-ignored]
├── models/                      # Local model weights              [git-ignored]
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

### 3. Set your model path

Edit the `_DEFAULT_MODEL_PATH` constant in `query_engine.py`:

```python
_DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
# or a local directory: "E:/models/my-llama"
```

### 4. Run

```bash
python query_engine.py
```

The first run ingests and indexes your documents. Subsequent runs skip unchanged files automatically.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | [LlamaIndex](https://www.llamaindex.ai/) 0.10+ |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (persistent) |
| Embedding Model | [BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3) (CPU, local) |
| LLM | Llama 3 (4-bit NF4 via `bitsandbytes`) |
| Sparse Retrieval | BM25 (`llama-index-retrievers-bm25`) |
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