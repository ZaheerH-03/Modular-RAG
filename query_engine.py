import os
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer, Settings

# Import your custom modules based on the new directory structure
from ingestion.build_vector_index import build_vector_index
from retrieval.hybrid_retriever import get_hybrid_rrf_retriever
from llm_loaders.local_llm_loader import QuantizedLocalLLM
from prompts.base import get_custom_prompt


def setup_query_engine(year: int, branch: str, model_path: str):
    print(f"Initializing Query Engine for {year} {branch.upper()}...")

    # 1. Load the LLM FIRST and register it globally.
    #    This MUST happen before any LlamaIndex retriever/synthesizer is built,
    #    otherwise QueryFusionRetriever's __init__ will call resolve_llm("default")
    #    and crash trying to reach OpenAI.
    print("Loading Local LLM...")
    llm = QuantizedLocalLLM(
        model_name=model_path,
        max_new_tokens=256,       # Reduced from 512 — limits generation VRAM
        context_window=2048,      # Reduced from 4096 — halves the KV cache footprint
        temperature=0.1
    )
    Settings.llm = llm  # Prevent LlamaIndex from falling back to OpenAI anywhere

    # 2. Fetch the synced index (handles all hashing and ingestion logic)
    index = build_vector_index(year, branch)

    # 3. Initialize the Hybrid RRF Retriever
    hybrid_retriever = get_hybrid_rrf_retriever(index)

    # 4. Load the Custom Strict Prompt
    qa_prompt = get_custom_prompt()

    # 5. Assemble the Synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt
    )

    # 6. Build the Final Query Engine
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer
    )

    return query_engine



if __name__ == "__main__":
    # Define your actual local model path here
    # Example: "meta-llama/Meta-Llama-3-8B-Instruct" or a local directory path
    LOCAL_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"

    engine = setup_query_engine(4, "ds", LOCAL_MODEL_PATH)

    test_query = "Different types of organizational behaviour models?"
    print(f"\nUser Query: {test_query}\n")

    response = engine.query(test_query)

    print("\n" + "=" * 50)
    print("Answer:")
    print("=" * 50)
    print(response.response)
    print("\n" + "=" * 50)

    print("SOURCES USED:")
    for node in response.source_nodes:
        doc_name = os.path.basename(node.metadata.get('file_path', 'Unknown'))
        page = node.metadata.get('page_label', '?')
        print(f"- {doc_name}, p.{page} (Score: {node.score:.4f})")