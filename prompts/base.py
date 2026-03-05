"""
prompts/base.py
===============
Provides the custom strict QA prompt template used by the RAG response
synthesizer.
"""

from llama_index.core import PromptTemplate

from interfaces import BasePromptProvider

# ---------------------------------------------------------------------------
# Prompt template string
# ---------------------------------------------------------------------------

_QA_TEMPLATE: str = (
    "SYSTEM ROLE:\n"
    "You are an exam tutor. Answer questions ONLY using the provided sources.\n\n"
    "STRICT RULES:\n"
    "- Do NOT repeat the question, instructions, or sources.\n"
    "- Do NOT explain your reasoning process.\n"
    "- Do NOT mention the word \"source\" except in citations like [Source 1].\n"
    "- You MAY rephrase and summarize the information in your own words.\n"
    "- Do NOT copy sentences verbatim unless necessary.\n"
    "- When the question asks for \"types\", explicitly list and briefly explain each type using the sources.\n"
    "- If the answer is not present in the sources, reply EXACTLY with:\n"
    "\"I cannot answer this from the notes.\"\n\n"
    "SOURCES (read-only):\n"
    "<<<\n"
    "{context_str}\n"
    ">>>\n\n"
    "QUESTION:\n"
    "{query_str}\n\n"
    "FINAL ANSWER:\n"
)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class CustomPromptProvider(BasePromptProvider):
    """
    Provides the strict, source-grounded QA prompt used by the RAG pipeline.

    The prompt enforces that the LLM answers only from the retrieved context,
    cites sources inline, and replies with a fixed phrase when the context
    does not contain an answer.
    """

    def get_prompt(self) -> PromptTemplate:
        """
        Build and return the QA prompt template.

        Returns:
            A ``PromptTemplate`` with ``{context_str}`` and ``{query_str}``
            placeholders, ready for use as ``text_qa_template`` in a response
            synthesizer.
        """
        return PromptTemplate(_QA_TEMPLATE)


# ---------------------------------------------------------------------------
# Module-level convenience function (keeps existing call sites unbroken)
# ---------------------------------------------------------------------------


def get_custom_prompt() -> PromptTemplate:
    """
    Convenience wrapper — return the QA prompt via :class:`CustomPromptProvider`.

    Returns:
        A ``PromptTemplate`` for strict, source-grounded QA.
    """
    return CustomPromptProvider().get_prompt()