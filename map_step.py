"""Step 3: the "map" half of map-reduce — summarize each chunk independently."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from load_doc import PDF_PATH, load_pdf
from split_doc import split_docs

load_dotenv()

# thinking disabled: this is a straightforward summarization task, not one
# that needs Claude's extended reasoning — and it keeps the response a
# plain string instead of a list of content blocks.
llm = ChatAnthropic(model="claude-opus-5", max_tokens=1024, thinking={"type": "disabled"})

map_prompt = ChatPromptTemplate.from_template(
    "Summarize the following excerpt from a lecture in 2-3 sentences. "
    "Focus on the key concepts introduced, not minor examples.\n\n"
    "Excerpt:\n{text}"
)

# LCEL syntax: chain the prompt, the model, and an output parser together
# with "|". Each piece's output becomes the next piece's input.
map_chain = map_prompt | llm | StrOutputParser()


def summarize_chunks(chunks) -> list[str]:
    # .batch() sends all chunk requests concurrently instead of looping and
    # awaiting each one — much faster for independent, parallel-safe calls
    # like this (that's exactly what "map" means: same op, many inputs).
    inputs = [{"text": chunk.page_content} for chunk in chunks]
    return map_chain.batch(inputs)


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = split_docs(pages)

    summaries = summarize_chunks(chunks)

    for i, summary in enumerate(summaries):
        print(f"--- Chunk {i} summary ---")
        print(summary)
        print()
