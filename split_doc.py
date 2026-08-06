"""Step 2: split the loaded pages into smaller chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_doc import PDF_PATH, load_pdf


def split_docs(docs, chunk_size: int = 1500, chunk_overlap: int = 150):
    # RecursiveCharacterTextSplitter tries to cut on paragraph breaks first,
    # then sentences, then words — so chunks stay coherent instead of
    # slicing mid-sentence. chunk_overlap repeats a bit of text between
    # chunks so context isn't lost right at the boundary.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # split_documents() splits each input Document independently — with a
    # PDF loaded page-by-page, that means it never merges short pages
    # together. Join everything into one continuous text first so the
    # splitter can produce well-sized chunks regardless of page breaks.
    full_text = "\n\n".join(d.page_content for d in docs)
    return splitter.create_documents([full_text])


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = split_docs(pages)

    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print()
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content[:200])
        print()
