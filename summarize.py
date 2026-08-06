"""Document summarizer — map-reduce over a PDF using Claude via LangChain.

Usage:
    python3 summarize.py "/path/to/document.pdf"
"""

import sys

from load_doc import load_pdf
from map_step import summarize_chunks
from reduce_step import combine_summaries
from split_doc import split_docs


def summarize_pdf(path: str) -> str:
    pages = load_pdf(path)
    chunks = split_docs(pages)
    print(f"Loaded {len(pages)} pages, split into {len(chunks)} chunks...", file=sys.stderr)

    chunk_summaries = summarize_chunks(chunks)
    print(f"Summarized {len(chunk_summaries)} chunks, combining...", file=sys.stderr)

    return combine_summaries(chunk_summaries)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 summarize.py <path-to-pdf>", file=sys.stderr)
        sys.exit(1)

    final_summary = summarize_pdf(sys.argv[1])

    print()
    print(final_summary)
