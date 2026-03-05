"""
interfaces/base_loader.py
=========================
Abstract contract for document-loading and change-detection components.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Defines the interface that any document-loading strategy must satisfy.
    Concrete implementations are responsible for discovering files on disk,
    computing change hashes, and returning only the documents that have been
    added or modified since the last ingestion run.
    """

    @abstractmethod
    def get_document_updates(
        self,
        year: int,
        branch: str,
        db_dir: str,
    ) -> tuple[bool, list[Any], list[str], dict[str, str], str]:
        """
        Detect new, modified, and deleted documents for a given year/branch.

        Args:
            year:    Target year (cumulative — includes years 1 through ``year``).
            branch:  Branch identifier (e.g. ``"ds"``, ``"cs"``).
            db_dir:  Directory used to persist hash-state JSON files.

        Returns:
            A 5-tuple of:
            - ``needs_update``  (bool)  – True if any change was detected.
            - ``changed_docs``  (list)  – Loaded Document objects for new/modified files.
            - ``deleted_files`` (list)  – File paths removed since the last run.
            - ``current_hashes``(dict)  – Current ``{filepath: md5_hash}`` mapping.
            - ``state_log``     (str)   – Path to the JSON state file to persist.
        """
        ...
