import chromadb
import json
import os

from ingestion.data_loader import get_document_updates
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = "E:/Documents/data_Science/ModularRAG"
DB_DIR = os.path.join(BASE_DIR, "chromadb")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models/bge-m3")


def load_embedding_model(model_name: str):
    # Must be CPU: LLM loads onto GPU first (required for OpenAI fallback fix),
    # so bge-m3 must stay on CPU or it will consume remaining VRAM and cause
    # a CUDA OOM during the LLM's inference forward pass (RMSNorm).
    return HuggingFaceEmbedding(model_name=model_name, device="cpu")


def get_vector_db(year: int, branch: str) -> StorageContext:
    collection_name = f"{year}_yr_{branch}"
    db = chromadb.PersistentClient(path=DB_DIR)
    existing_collections = [c.name for c in db.list_collections()]
    if collection_name in existing_collections:
        collection = db.get_collection(collection_name)
        if collection.count() > 0:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            return StorageContext.from_defaults(vector_store=vector_store)
    chromadb_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chromadb_collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def build_vector_index(year: int, branch: str):
    needs_update, changed_docs, deleted_files, current_hashes, state_log = \
        get_document_updates(year, branch, DB_DIR)

    embedding_model = load_embedding_model(EMBEDDING_MODEL_PATH)
    storage_context = get_vector_db(year, branch)

    index = VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        embed_model=embedding_model
    )

    if needs_update:
        splitter = SemanticSplitterNodeParser(
            embed_model=embedding_model,
            buffer_size=3,
            breakpoint_percentile_threshold=95
        )

        if changed_docs:
            nodes = splitter.get_nodes_from_documents(changed_docs)
            index.insert_nodes(nodes)

        for fp in deleted_files:
            index.delete_ref_doc(fp, delete_from_docstore=True)

        with open(state_log, "w") as f:
            json.dump(current_hashes, f)

        print(f"Updated index: {len(changed_docs)} doc(s) added/modified, {len(deleted_files)} file(s) removed.")
    else:
        print("No need to update.")

    return index


if __name__ == "__main__":
    vector_index = build_vector_index(4, "ds")
    vectordb = chromadb.PersistentClient(path=DB_DIR)
    coll = vectordb.get_collection("4_yr_ds")
    print(f"Nodes in 4th_yr_ds: {coll.count()}")