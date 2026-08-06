"""Step 1: load a PDF and see what LangChain gives us back."""

from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "/Users/megibida/Downloads/15-Ein und Ausgabe.pdf"


def load_pdf(path: str):
    loader = PyPDFLoader(path)
    # .load() returns a list of Document objects — LangChain splits a PDF
    # into one Document per page automatically.
    return loader.load()


if __name__ == "__main__":
    docs = load_pdf(PDF_PATH)

    print(f"Loaded {len(docs)} pages")
    print()
    print("--- First page metadata ---")
    print(docs[0].metadata)
    print()
    print("--- First page content (first 500 chars) ---")
    print(docs[0].page_content[:500])
    print()
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Total characters across all pages: {total_chars}")
