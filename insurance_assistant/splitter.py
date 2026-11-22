import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

# Generalized regex pattern matching structural section boundaries in insurance and legal documents
SECTION_HEADER_PATTERN = re.compile(
    r'(?m)^[ \t]*(?:'
    r'#+\s+[^\n]+|'  # Markdown headings (# Title, ## Subtitle)
    r'[0-9]+\.\s+(?:DEFINITIONS?|SCOPE OF COVER|BENEFITS|EXCLUSIONS?|MAJOR EXCLUSIONS?|WAITING PERIODS?|CLAIMS? PROCEDURES?|CONDITIONS?)[ \t]*$|'  # Numbered major headers
    r'(?:DEFINITIONS|SCOPE OF COVER|BENEFITS|EXCLUSIONS|MAJOR EXCLUSIONS|WAITING PERIODS|General Conditions|GENERAL CONDITIONS)[ \t]*$|'  # Standalone uppercase section titles
    r'[a-z]\)\s+[A-Za-z0-9\s/&\-]{3,40}(?::|\n|$)|'  # Lettered subsections e.g. a) Due Care, f) Claims Procedures, p) Geographical limits:
    r'Annexure\s+[A-Za-z0-9\s\-]+'  # Annexures and appendices
    r')',
    re.IGNORECASE,
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
