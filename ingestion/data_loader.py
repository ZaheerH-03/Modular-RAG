import os
import hashlib
import json
from llama_index.core import SimpleDirectoryReader

BASE_DIR = "E:/Documents/data_Science/ModularRAG"
_YEAR_PREFIXES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def _compute_hash_state(current_year: int, branch: str) -> dict:
    """Walks all cumulative year directories and returns a {filepath: md5_hash} dict."""
    current_state = {}
    for y in range(1, current_year + 1):
        dir_path = os.path.join(BASE_DIR, "data", f"{_YEAR_PREFIXES[y]}_yr_{branch}")
        if not os.path.exists(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                hasher = hashlib.md5()
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
                current_state[file_path] = hasher.hexdigest()
    return current_state


def get_document_updates(year: int, branch: str, db_dir: str):
    """
    Checks for file additions, modifications, and deletions across all cumulative
    year directories for the given branch. Loads only the changed/new documents.

    Args:
        year:    The target year (cumulative — includes years 1 through `year`).
        branch:  The branch name (e.g. "ds", "cs").
        db_dir:  Path to the ChromaDB directory (used to store hash state files).

    Returns:
        needs_update  (bool):  True if any change was detected.
        changed_docs  (list):  Loaded Document objects for new/modified files only.
        deleted_files (list):  File paths removed since last run (delete from index).
        current_hashes (dict): Current {filepath: hash} state — save after indexing.
        state_log     (str):   Path to the JSON state file to write current_hashes to.
    """
    os.makedirs(db_dir, exist_ok=True)
    state_log = os.path.join(db_dir, f"{year}_{branch}_hash_state.json")

    # Load previous state
    old_state = {}
    if os.path.exists(state_log):
        with open(state_log, "r") as f:
            old_state = json.load(f)

    # Compute current state
    current_hashes = _compute_hash_state(year, branch)

    # Detect changes
    changed_files = [
        fp for fp, fhash in current_hashes.items()
        if fp not in old_state or old_state[fp] != fhash
    ]
    deleted_files = [fp for fp in old_state if fp not in current_hashes]
    needs_update = bool(changed_files or deleted_files)

    # Load only changed/new documents
    changed_docs = []
    if changed_files:
        loader = SimpleDirectoryReader(input_files=changed_files, filename_as_id=True)
        changed_docs = loader.load_data(show_progress=True)

    return needs_update, changed_docs, deleted_files, current_hashes, state_log