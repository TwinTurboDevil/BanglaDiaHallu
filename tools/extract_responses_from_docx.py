#!/usr/bin/env python3
"""
Convert the response document ("Model Responses.docx") into a machine-readable CSV.

The document has one section per configuration. Each section starts with the
configuration name and a short "Parameters" block, followed by the 40 questions.
Every question (a paragraph that starts with a Bangla numeral, e.g. "১. ...") is
followed by a two-row table whose rows hold "Answer 1" and "Answer 2", the two
independent generations (iterations 1 and 2).

Output columns
--------------
config, question, iteration, domain, question_text_recorded, response

The response text is copied without any editing; only leading and trailing
whitespace is removed. Run from the repository root:

    python tools/extract_responses_from_docx.py \
        --docx "data/original_documents/Model Responses.docx" \
        --out data/model_responses.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import docx  # python-docx
from docx.table import Table
from docx.text.paragraph import Paragraph

# Section headings in the document -> configuration names used in the paper
SECTION_TO_CONFIG = {
    "Qwen 3.5 9B": "Qwen3.5-9B",
    "TigerLLM": "TigerLLM-9B",
    "Qwen 3.6 35B A3B": "Qwen3.6-35B-A3B",
    "Gemma 4 26B A4B": "Gemma-4-26B-A4B",
    "Qwen 3.5 9B + RAG": "BanglaDiaRAG",
}
DOMAINS = [("Dietary", 1, 9), ("Medication and insulin", 10, 16), ("Lifestyle and exercise", 17, 21),
           ("Complications and safety", 22, 26), ("Symptoms and early detection", 27, 32),
           ("Special situations", 33, 40)]
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
QUESTION_RE = re.compile(r"^([০-৯]+)\.\s*(.+)$", re.S)


def domain_of(q: int) -> str:
    return next(name for name, lo, hi in DOMAINS if lo <= q <= hi)


def iter_blocks(document):
    """Yield ('p', text) and ('t', Table) in document order."""
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text
            if text.strip():
                yield "p", text
        elif tag == "tbl":
            yield "t", Table(child, document)


def extract(docx_path: Path) -> list[dict]:
    document = docx.Document(str(docx_path))
    rows, config, question, qtext = [], None, None, None
    for kind, item in iter_blocks(document):
        if kind == "p":
            text = item.replace("​", "").strip()
            if text in SECTION_TO_CONFIG:
                config, question = SECTION_TO_CONFIG[text], None
                continue
            m = QUESTION_RE.match(text)
            if m and config:
                question, qtext = int(m.group(1).translate(BN_DIGITS)), m.group(2).strip()
            continue
        if config is None or question is None:
            continue
        for r in item.rows:
            label, answer = r.cells[0].text.strip(), r.cells[1].text.strip()
            it = int(label.split()[-1])  # "Answer 1" / "Answer 2"
            rows.append(dict(config=config, question=question, iteration=it, domain=domain_of(question),
                             question_text_recorded=qtext, response=answer))
        question = None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", default="data/original_documents/Model Responses.docx")
    ap.add_argument("--out", default="data/model_responses.csv")
    args = ap.parse_args()
    rows = extract(Path(args.docx))
    # 5 configurations x 40 questions x 2 generations
    assert len(rows) == 400, len(rows)
    keys = {(r["config"], r["question"], r["iteration"]) for r in rows}
    assert len(keys) == 400, "duplicate (config, question, iteration)"
    order = list(SECTION_TO_CONFIG.values())
    rows.sort(key=lambda r: (order.index(r["config"]), r["question"], r["iteration"]))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} responses to {args.out}")


if __name__ == "__main__":
    main()
