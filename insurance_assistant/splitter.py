import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

# Generalized regex pattern matching structural section boundaries in contracts and legal documents
SECTION_HEADER_PATTERN = re.compile(
    r'(?m)^[ \t]*(?:'
    r'#{1,6}\s+[^\n]+|'  # Markdown headings (# Title, ## Subtitle)
    r'(?:Section|Article|Part|Chapter|Clause)\s+[0-9IVXLCDM]+[.:\s\t]+[^\n]+|'  # Formal structural markers (Section 1, Article II)
    r'[0-9]+\.\s+[A-Z\s/&\-]{4,60}[ \t]*$|'  # Numbered ALL-CAPS major section headers (e.g. 1. TERMS OF COVER)
    r'(?:[A-Z][A-Z\s/&\-]{3,50}[A-Z]|General Conditions)[ \t]*$|'  # Standalone uppercase section titles (DEFINITIONS, EXCLUSIONS, etc.)
    r'[a-z]\)\s+[A-Za-z0-9\s/&\-]{3,40}(?::|\n|$)|'  # Lettered legal subsections (a) Due Care, p) Geographical limits:)
    r'(?:Annexure|Schedule|Appendix|Exhibit)\s+[A-Za-z0-9\s\-]+'  # Annexures, Schedules, Appendices
    r')'
)


def split_into_chunks(documents: list) -> list:
    """
    Split documents into chunks using header-aware structural sectioning.
    
    Identifies document section boundaries and prepends hierarchical section
    context ([SECTION: <SECTION_NAME>]) to each chunk to prevent context inversion
    errors (e.g. clauses under EXCLUSIONS being misread as affirmative benefits).
    """
    logger.info(
        "Splitting %d document(s) with Header-Aware Chunking (size=%d, overlap=%d)",
        len(documents),
        config.chunk_size,
        config.chunk_overlap,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    all_chunks = []

    for doc in documents:
        text = doc.page_content
        matches = list(SECTION_HEADER_PATTERN.finditer(text))

        # Fallback to standard recursive splitting if no structural headers are detected
        if not matches:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            continue

        prev_end = 0
        prev_title = "PREAMBLE"
        sections = []

        for m in matches:
            start = m.start()
            content = text[prev_end:start].strip()
            if content:
                sections.append((prev_title, content))
            prev_title = m.group(0).strip().split("\n")[0].strip().rstrip(":")
            prev_end = m.end()

        trailing_content = text[prev_end:].strip()
        if trailing_content:
            sections.append((prev_title, trailing_content))

        for sec_title, sec_text in sections:
            sec_doc = Document(
                page_content=sec_text,
                metadata={**doc.metadata, "section": sec_title},
            )
            sub_chunks = text_splitter.split_documents([sec_doc])
            for sc in sub_chunks:
                sc.page_content = f"[SECTION: {sec_title}]\n" + sc.page_content
                all_chunks.append(sc)

    logger.info("Created %d header-aware chunks across document(s)", len(all_chunks))
    return all_chunks
