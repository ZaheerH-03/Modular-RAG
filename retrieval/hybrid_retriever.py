from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

def get_hybrid_rrf_retriever(index):
    # 1. Vector Retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )

    # 2. BM25 Retriever
    # Attempt to fetch nodes from the docstore first
    nodes = list(index.docstore.docs.values())
    
    # FIX: If the docstore is empty, dynamically reconstruct nodes directly from ChromaDB
    if not nodes:
        print("Docstore is empty. Reconstructing text nodes from ChromaDB for BM25...")
        
        # Access the underlying Chroma collection tied to the index
        collection = index.storage_context.vector_store._collection
        
        # Extract the stored text and metadata directly
        chroma_data = collection.get(include=["documents", "metadatas"])
        
        # Rebuild the LlamaIndex TextNode objects
        for doc_id, text, metadata in zip(chroma_data["ids"], chroma_data["documents"], chroma_data["metadatas"]):
            nodes.append(TextNode(id_=doc_id, text=text, metadata=metadata or {}))
            
        if not nodes:
            raise ValueError("ChromaDB is completely empty. Ensure your ingestion pipeline has processed documents.")

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )

    # 3. RRF Fusion Retriever
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=2,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )

    return retriever