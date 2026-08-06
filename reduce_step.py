"""Step 4: the "reduce" half of map-reduce — combine chunk summaries into one."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from load_doc import PDF_PATH, load_pdf
from map_step import summarize_chunks
from split_doc import split_docs

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=4096, thinking={"type": "disabled"})

reduce_prompt = ChatPromptTemplate.from_template(
    "The following are summaries of consecutive sections of a lecture, in order. "
    "Combine them into a single, coherent summary of the whole lecture. "
    "Do not just list the sections — synthesize them into flowing prose that "
    "captures the overall narrative and key takeaways.\n\n"
    "Section summaries:\n{summaries}"
)

reduce_chain = reduce_prompt | llm | StrOutputParser()


def combine_summaries(summaries: list[str]) -> str:
    joined = "\n\n".join(f"{i + 1}. {s}" for i, s in enumerate(summaries))
    return reduce_chain.invoke({"summaries": joined})


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = split_docs(pages)
    chunk_summaries = summarize_chunks(chunks)

    final_summary = combine_summaries(chunk_summaries)

    print("--- Final summary ---")
    print(final_summary)
