"""
ingestion/data_loader.py
========================
Handles document discovery, MD5 hash-based change detection, and incremental
loading of only new or modified files from cumulative year directories.
"""

import hashlib
import json
import os
from typing import Any

from llama_index.core import SimpleDirectoryReader

from config import cfg
from interfaces import BaseDocumentLoader

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASE_DIR: str = cfg.paths.base_dir

_YEAR_PREFIXES: dict[int, str] = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_hash_state(current_year: int, branch: str) -> dict[str, str]:
    """
    Walk all cumulative year directories and return a ``{filepath: md5_hash}``
    mapping for every file found.

    Args:
        current_year: The upper bound year (inclusive).
        branch:       Branch identifier (e.g. ``"ds"``, ``"cs"``).

    Returns:
        A dictionary mapping absolute file paths to their MD5 hex digests.
    """
    current_state: dict[str, str] = {}

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


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class IncrementalDocumentLoader(BaseDocumentLoader):
    """
    Concrete document loader that uses MD5 hashing to detect and load only
    new or modified files since the last ingestion run.

    Change state is persisted as a JSON file inside ``db_dir`` so that
    subsequent runs can compare the current file tree against the previous
    snapshot.
    """

    def get_document_updates(
        self,
        year: int,
        branch: str,
        db_dir: str,
    ) -> tuple[bool, list[Any], list[str], dict[str, str], str]:
        """
        Detect new, modified, and deleted documents for the given year/branch.

        Args:
            year:    Target year (cumulative — includes years 1 through ``year``).
            branch:  Branch identifier (e.g. ``"ds"``, ``"cs"``).
            db_dir:  Directory used to persist hash-state JSON files.

        Returns:
            A 5-tuple of:
            - ``needs_update``   (bool) – True when any change was detected.
            - ``changed_docs``   (list) – Loaded ``Document`` objects for new/modified files.
            - ``deleted_files``  (list) – File paths removed since the last run.
            - ``current_hashes`` (dict) – Current ``{filepath: md5_hash}`` snapshot.
            - ``state_log``      (str)  – Path to the JSON state file to save.
        """
        os.makedirs(db_dir, exist_ok=True)
        state_log = os.path.join(db_dir, f"{year}_{branch}_hash_state.json")

        # Load previous state ---------------------------------------------------
        old_state: dict[str, str] = {}
        if os.path.exists(state_log):
            with open(state_log, "r") as f:
                old_state = json.load(f)

        # Compute current state -------------------------------------------------
        current_hashes = _compute_hash_state(year, branch)

        # Detect additions / modifications / deletions --------------------------
        changed_files = [
            fp for fp, fhash in current_hashes.items()
            if fp not in old_state or old_state[fp] != fhash
        ]
        deleted_files = [fp for fp in old_state if fp not in current_hashes]
        needs_update = bool(changed_files or deleted_files)

        # Load only new / changed documents -------------------------------------
        changed_docs: list[Any] = []
        if changed_files:
            loader = SimpleDirectoryReader(
                input_files=changed_files,
                filename_as_id=True,
            )
            changed_docs = loader.load_data(show_progress=True)

        return needs_update, changed_docs, deleted_files, current_hashes, state_log


# ---------------------------------------------------------------------------
# Module-level convenience function (keeps existing call sites unbroken)
# ---------------------------------------------------------------------------


def get_document_updates(
    year: int,
    branch: str,
    db_dir: str,
) -> tuple[bool, list[Any], list[str], dict[str, str], str]:
    """
    Convenience wrapper around :class:`IncrementalDocumentLoader`.

    Args:
        year:    Target year (cumulative — includes years 1 through ``year``).
        branch:  Branch identifier (e.g. ``"ds"``, ``"cs"``).
        db_dir:  Directory used to persist hash-state JSON files.

    Returns:
        See :meth:`IncrementalDocumentLoader.get_document_updates`.
    """
    return IncrementalDocumentLoader().get_document_updates(year, branch, db_dir)