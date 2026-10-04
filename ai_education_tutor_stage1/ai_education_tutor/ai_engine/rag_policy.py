import os
from pypdf import PdfReader


POLICY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "RAG_POLICY.pdf"
)


DEFAULT_POLICY = """
STRICT RAG POLICY

1. Do not invent facts.
2. Do not guess missing information.
3. For PDF mode, use ONLY the supplied PDF evidence.
4. Do not use outside knowledge in PDF mode.
5. Do not fabricate examples, definitions, formulas,
   dates, names, statistics, references, citations,
   quotations, page numbers, or URLs.
6. If the PDF does not contain enough information,
   clearly state that the information was not found
   in the provided PDF.
"""


PDF_FALLBACK = (
    "I couldn't find this information in the provided PDF."
)


def load_rag_policy():
    """
    Load the RAG policy from RAG_POLICY.pdf.
    """

    if not os.path.exists(POLICY_FILE):
        return DEFAULT_POLICY

    try:
        reader = PdfReader(POLICY_FILE)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        policy = "\n".join(pages).strip()

        if policy:
            return policy

    except Exception:
        pass

    return DEFAULT_POLICY


RAG_POLICY = load_rag_policy()
