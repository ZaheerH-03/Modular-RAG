import chromadb
from data_loader import get_data_loader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex , StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

embedding_model_name = "E:/Documents/data_Science/ModularRAG/models/bge-m3"
def load_embedding_model(model_name):
    embedding_model = HuggingFaceEmbedding(model_name)
    return embedding_model

def get_vector_db(year: int, branch: str):
    db = chromadb.PersistentClient(path = "./chromadb")
    chromadb_collection = db.get_or_create_collection(f"{year}_yr_{branch}")
    vector_store = ChromaVectorStore(chroma_collection=chromadb_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def build_vector_index(year: int, branch: str):
    embedding_model = load_embedding_model(embedding_model_name)
    loader = get_data_loader(year, branch)
    documents = loader.load_data(show_progress=True)
    splitter = SemanticSplitterNodeParser(embed_model=embedding_model,buffer_size=3,breakpoint_percentile_threshold=95)
    nodes = splitter.get_nodes_from_documents(documents)
    storage_context = get_vector_db(year, branch)
    index = VectorStoreIndex(storage_context=storage_context,nodes=nodes,embed_model=embedding_model)
    return index

if __name__ == "__main__":
    vector_index = build_vector_index(4,"ds")
    vectordb = chromadb.PersistentClient(path="./chromadb")
    coll = vectordb.get_collection("4_yr_ds")
    print(f"Nodes in 4th_yr_ds: {coll.count()}")

