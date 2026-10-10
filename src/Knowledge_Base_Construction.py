#!/usr/bin/env python3
"""
BanglaDiaRAG - knowledge base construction (paper: Methods, "Knowledge base
construction"; Fig 2A).

Input
-----
A Markdown file with the curated, physician-reviewed Bangla text
(default: data/knowledge_base/bangla_guidelines_new.md). The file was prepared
manually: passages relevant to the 40 questions were selected from the National
Guideline on Diabetes Mellitus (2023) and a few supplementary sources, translated
into formal Bangla with Gemini 3.1 Pro and reviewed by the two physicians. Each section
starts with a bold header at the beginning of a line (**...**) and contains one
source tag of the form

    +++ [National Guidelines on Diabetes Mellitus 2023, Chapter 3: Medical Nutrition Therapy]

(an escaped "+++ \\[...\\]" and a leading "Source:" are also accepted).

Processing (identical to the version used in the study)
--------------------------------------------------------
1. Split the file at the bold headers.
2. Move the source tag of each section into the chunk metadata and remove it
   from the text.
3. Split long sections with LangChain's RecursiveCharacterTextSplitter:
   chunk_size = 1,200 characters, chunk_overlap = 100 characters, separators
   (in order of preference) = blank lines, newline, Bangla danda "।", space.
4. Prefix every chunk with its header: "বিষয়: <header>\\nতথ্য: <text>"
   ("topic" / "information"), so that each chunk is interpretable on its own.
5. Embed the chunks with BAAI/bge-m3 on the CPU (normalize_embeddings=True,
   1,024 dimensions) and store them in a persistent Chroma collection named
   "bangla_diabetes_guidelines". The collection uses Chroma's default "l2"
   space; because the vectors have unit length, squared L2 distance equals
   2 - 2 * cosine similarity, so the ranking equals cosine ranking (paper, Eq 1).

The sparse BM25 index is not stored: BanglaDiaRAG_Pipeline.py rebuilds it at
start-up from the same chunks read back from Chroma.

Additions to the original script (they do not change the knowledge base)
-------------------------------------------------------------------------
* command-line arguments instead of hard-coded paths;
* a refusal to write into a non-empty collection unless --overwrite is given
  (the original script silently added a second copy of every chunk when it was
  run twice);
* the chunk IDs are used as Chroma record IDs, so a rebuild is idempotent;
* an export of all chunks and their metadata (chunks.jsonl) and summary
  statistics (kb_stats.json) for reporting and auditing.

Usage
-----
    python src/Knowledge_Base_Construction.py \\
        --kb-file data/knowledge_base/bangla_guidelines_new.md \\
        --db-dir ./chroma_bangla_db --export-dir results/knowledge_base
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "bangla_diabetes_guidelines"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1200          # characters
CHUNK_OVERLAP = 100        # characters
SEPARATORS = ["\n\n\n\n", "\n\n", "\n", "।", " ", ""]
SOURCE_PREFIX = "DIAB_GUIDE_BN"
HEADER_SPLIT_RE = re.compile(r"(?m)^(?=\*\*.*?\*\*)")
HEADER_RE = re.compile(r"\*\*(.*?)\*\*")
SOURCE_TAG_RE = re.compile(r"\+\+\+\s*\\?\[(.*?)\]")


# ---------------------------------------------------------------------------
# Header-based chunking and metadata extraction
# ---------------------------------------------------------------------------
def process_markdown_guideline(file_path: str, source_prefix: str = SOURCE_PREFIX) -> list[Document]:
    """Split the curated Markdown file into chunks with source metadata."""
    print(f"Reading {file_path} ...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as fh:
        markdown_text = fh.read()
    markdown_text = markdown_text.replace("\r\n", "\n")  # Windows line endings

    # Split directly at bold headers that start a line.
    raw_blocks = HEADER_SPLIT_RE.split(markdown_text.strip())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    final_documents = []
    for idx, block in enumerate(raw_blocks):
        block = block.strip()
        if not block:
            continue

        header_match = HEADER_RE.match(block)
        if header_match:
            header_name = header_match.group(1).strip()
            raw_content = block[header_match.end():].strip()
        else:
            header_name = "General Topic"
            raw_content = block

        source_match = SOURCE_TAG_RE.search(raw_content)
        if source_match:
            raw_source = source_match.group(1).strip()
            source_name = raw_source[7:].strip() if raw_source.lower().startswith("source:") else raw_source
            cleaned_content = SOURCE_TAG_RE.sub("", raw_content).strip()
        else:
            source_name = f"{source_prefix} General"
            cleaned_content = raw_content

        for sub_idx, sub_chunk in enumerate(text_splitter.split_text(cleaned_content)):
            chunk_id = f"{source_prefix}_P{idx + 1}_C{sub_idx + 1}_BN"
            final_documents.append(Document(
                page_content=f"বিষয়: {header_name}\nতথ্য: {sub_chunk.strip()}",
                metadata={"chunk_id": chunk_id, "source": source_name,
                          "topic": header_name, "category": source_prefix},
            ))

    print(f"Created {len(final_documents)} chunks with metadata.")
    return final_documents


# ---------------------------------------------------------------------------
# Dense index: BGE-M3 embeddings in a persistent Chroma collection (CPU)
# ---------------------------------------------------------------------------
def load_embeddings(device: str = "cpu") -> HuggingFaceBgeEmbeddings:
    # The LangChain BGE wrapper prepends its default English retrieval
    # instruction ("Represent this question for searching relevant passages: ")
    # to queries only; documents are embedded without an instruction.
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def store_in_vector_db(documents: list[Document], db_directory: str = "./chroma_bangla_db",
                       overwrite: bool = False, device: str = "cpu") -> Chroma:
    print(f"Loading {EMBEDDING_MODEL} on [{device}] ...")
    embeddings = load_embeddings(device)

    existing = Chroma(persist_directory=db_directory, embedding_function=embeddings,
                      collection_name=COLLECTION_NAME)
    n_existing = len(existing.get(include=[])["ids"])
    if n_existing:
        if not overwrite:
            raise SystemExit(f"Collection '{COLLECTION_NAME}' in {db_directory} already holds {n_existing} "
                             f"chunks. Use --overwrite to rebuild it.")
        existing.delete_collection()

    print(f"Embedding {len(documents)} chunks (this can take several minutes on a CPU) ...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=[d.metadata["chunk_id"] for d in documents],
        persist_directory=db_directory,
        collection_name=COLLECTION_NAME,
    )
    print(f"Knowledge base stored in {db_directory} (collection '{COLLECTION_NAME}').")
    return vector_store


# ---------------------------------------------------------------------------
# Export for auditing and reporting
# ---------------------------------------------------------------------------
def export_chunks(documents: list[Document], kb_file: str, export_dir: str) -> dict:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for d in documents:
            fh.write(json.dumps({"text": d.page_content, **d.metadata}, ensure_ascii=False) + "\n")
    raw = Path(kb_file).read_text(encoding="utf-8")
    lengths = sorted(len(d.page_content) for d in documents)
    stats = {
        "kb_file": str(kb_file),
        "words_whitespace_delimited": len(raw.split()),
        "characters": len(raw),
        "n_chunks": len(documents),
        "chunk_characters_min": lengths[0],
        "chunk_characters_median": lengths[len(lengths) // 2],
        "chunk_characters_max": lengths[-1],
        "chunks_per_source": dict(Counter(d.metadata["source"] for d in documents).most_common()),
        "settings": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "separators": SEPARATORS,
                     "embedding_model": EMBEDDING_MODEL, "normalize_embeddings": True,
                     "collection": COLLECTION_NAME, "distance": "l2 (Chroma default)"},
    }
    (out / "kb_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(documents)} chunks and statistics to {out}/")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb-file", default="data/knowledge_base/bangla_guidelines_new.md")
    ap.add_argument("--db-dir", default="./chroma_bangla_db")
    ap.add_argument("--export-dir", default="results/knowledge_base")
    ap.add_argument("--device", default="cpu", help="device for BGE-M3 (the study used the CPU)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild an existing collection")
    ap.add_argument("--no-index", action="store_true", help="only chunk and export; do not embed")
    args = ap.parse_args()

    docs = process_markdown_guideline(args.kb_file, SOURCE_PREFIX)
    if not docs:
        raise SystemExit("No chunks were created. Check the path of the Markdown file.")
    export_chunks(docs, args.kb_file, args.export_dir)
    if not args.no_index:
        store_in_vector_db(docs, args.db_dir, overwrite=args.overwrite, device=args.device)
        print("Done. Run src/BanglaDiaRAG_Pipeline.py next.")


if __name__ == "__main__":
    main()
