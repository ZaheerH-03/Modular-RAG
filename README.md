# 🎓 Scholarly-RAG: Multi-Tier Academic Assistant

**Scholarly-RAG** is a specialized Retrieval-Augmented Generation (RAG) system designed for college ecosystems. It implements a **Modular Silo Architecture** that segregates academic data by branch (DS, AI, CSE) and year, providing students with context-aware answers based on their academic level.



## 🚀 Key Features

- **Modular Silo Architecture**: Stores data in separate ChromaDB collections following the `year_yr_branch` format.
- **Cumulative Access Logic**: Implements a hierarchical retrieval system where 4th-year students can access 1st-3rd year data, while junior access remains restricted to their level.
- **Semantic Chunking**: Utilizes `SemanticSplitterNodeParser` to break text based on "topic shifts" rather than arbitrary character limits, ensuring mathematical and logical context is preserved.
- **Local Embedding Support**: Powered by the **BAAI/BGE-M3** model loaded from local paths, with full support for CUDA/GPU acceleration.
- **Persistent Storage**: Uses ChromaDB to save embeddings to disk, eliminating the need for re-processing documents during every session.

## 🛠️ Tech Stack

- **Framework**: LlamaIndex (v0.10+)
- **Vector Database**: ChromaDB
- **Embedding Model**: BAAI/BGE-M3 (HuggingFace)
- **Language**: Python 3.10+
- **Acceleration**: PyTorch with CUDA support

## 📁 Project Structure

```text
ModularRAG/
├── data/                 # Raw PDF files organized by year (Ignored in Git)
│   ├── 1st_yr_ds/
│   ├── 4th_yr_ds/
├── ingestion/            # Core data processing logic
│   ├── data_loader.py    # Recursive directory and cumulative path logic
│   └── build_index.py    # Splitting, Embedding, and Vector Storage
├── models/               # Local model weights/snapshots (Ignored in Git)
├── chromadb/             # Persistent Vector Store directory (Ignored in Git)
├── requirements.txt      # Python dependencies
└── README.md