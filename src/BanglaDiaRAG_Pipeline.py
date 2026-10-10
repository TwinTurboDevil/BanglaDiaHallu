#!/usr/bin/env python3
"""
BanglaDiaRAG - guideline-grounded hybrid retrieval-augmented generation
(paper: Methods, "Retrieval and generation", "Prompting strategy",
"Decoding temperature"; Fig 2B).

For every question the pipeline
  1. retrieves the top 3 chunks with BM25 (whitespace tokens, k1 = 1.5, b = 0.75)
     and the top 3 chunks with BGE-M3 dense retrieval from Chroma (paper, Eqs 1-2);
  2. merges them, BM25 results first, and removes duplicate chunks, giving 3-6
     context chunks (paper, Eq 3);
  3. fills a ChatML prompt: system prompt, then CONTEXT and QUESTION in one user
     turn, and an assistant turn that is pre-filled with a closed reasoning block
     ("<think>\\nThought process skipped.\\n</think>") so that Qwen3.5 answers
     without reasoning (zero-shot; no worked examples);
  4. generates the answer with Qwen3.5-9B (GGUF, Q4_K_M) through llama.cpp at
     temperature 0.5, context window 8,192 tokens and at most 8,192 new tokens,
     without a fixed seed;
  5. removes any <think>...</think> block from the output and appends a source
     list ("সূত্র") built from the metadata of the retrieved chunks, so the model
     cannot invent citations.

Sampling parameters other than temperature are the defaults of LangChain's
LlamaCpp wrapper (top_p = 0.95, top_k = 40, repeat_penalty = 1.1).

RoPE base frequency: LangChain's LlamaCpp wrapper passes rope_freq_base = 10000
to llama.cpp unless another value is given, and this overrides the value stored
in the GGUF file. The model configuration of Qwen3.5-9B specifies a RoPE base
(rope_theta) of 10,000,000 for its full-attention layers (Hugging Face model
card, Qwen/Qwen3.5-9B). This script therefore defaults to --rope-freq-base 0,
which makes llama.cpp use the native value stored in the GGUF file. The value
actually used is saved in the run metadata (rope_freq_base).

Usage
-----
    python src/BanglaDiaRAG_Pipeline.py --model models/Qwen3.5-9B-Q4_K_M.gguf \\
        --db-dir ./chroma_bangla_db --runs 2 --out results/banglaDiaRAG
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

COLLECTION_NAME = "bangla_diabetes_guidelines"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 3                 # per retriever
TEMPERATURE = 0.5
N_CTX = 8192
MAX_TOKENS = 8192
STOP = ["<|im_end|>", "<|endoftext|>"]

# ---------------------------------------------------------------------------
# System prompt 
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are a professional and empathetic medical AI assistant dedicated to helping diabetic patients in Bangladesh.

STRICT RULES:
Language: Respond in a natural mix of Bangla and English (Code-mixed Bangla), as a doctor would speak to a patient.
Tone: Be supportive, clear, and culturally respectful.
Medical Terms: Use Bengali script for common medical terms (e.g., use 'ইনসুলিন', 'ব্লাড সুগার', 'ডায়েট').
GROUNDING: Answer strictly based on the CONTEXT provided. If the CONTEXT does not contain the answer, say: "এই প্রশ্নের উত্তরের জন্য দয়া করে আপনার ডাক্তারের পরামর্শ নিন।"
NO REFERENCES IN TEXT: Do not write source names in your response.
Safety: If a question is dangerous or outside your knowledge, advise the patient to see a doctor immediately.
DISCLAIMER: End every response with a short final section that begins with the exact heading "সতর্কীকরণ:". In it, say that you are an AI assistant, that this advice is for general information only and is not a substitute for direct clinical advice from a doctor, and that the patient should consult their doctor before making any health decision."""




# ChatML template; the assistant turn is pre-filled with a closed reasoning block.
PROMPT_TEMPLATE = """<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
### CONTEXT:
{context}

### QUESTION:
{question}<|im_end|>
<|im_start|>assistant
<think>
Thought process skipped.
</think>
"""

# Integrity check for the system prompt and template
SYSTEM_INSTRUCTION_SHA256 = "7b6722464d510abbd768eb94d8083e5d4c11fe80d097906d3208de3c0b6bb14d"
PROMPT_TEMPLATE_SHA256 = "4590c3142e3d4c5d4b2afd403a100fc4818c719bfca3c62d4d5d5357d768343c"
assert hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest() == SYSTEM_INSTRUCTION_SHA256
assert hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest() == PROMPT_TEMPLATE_SHA256

# The 40 questions exactly as submitted to BanglaDiaRAG in the study;
#see data/questions.csv and S1 Table).
QUESTIONS = [
    "ডায়াবেটিস হলে কি আমি দিনে তিন বেলা ভাত খেতে পারব নাকি ভাতের বদলে রুটি খাওয়া যাবে?",
    "ডায়াবেটিস হলে কি আমার জন্য মিষ্টি ফল (যেমন: আম, কাঁঠাল) খাওয়া একদমই নিষেধ, নাকি নির্দিষ্ট পরিমাণে খাওয়া যাবে?",
    "ডায়াবেটিস হলে কি আমি চিনি ছাড়া চা খেতে পারবো আর খেতে পারলে দিনে কত কাপ খেতে পারবো?",
    "ডায়াবেটিস রোগীর জন্য কি মধু বা গুড় চিনির বিকল্প হিসেবে ব্যবহার করা নিরাপদ?",
    "ডায়াবেটিস হলে কি আমি মিষ্টি খেতে পারবো?",
    "ডায়াবেটিস রোগীর জন্য কি কি শাক-সবজি খাওয়া ভালো?",
    "ডায়াবেটিসের রোগীরা ইফতারে খেজুর বা শরবত খেতে পারবে কি?",
    "ডায়াবেটিস হলে আমি কি কি খাবো আর কি কি এড়িয়ে চলবো?",
    "ডায়াবেটিস রোগীর জন্য ডায়াবেটিস ডায়েট-চার্ট কেমন হওয়া উচিত?",
    "আমি ডায়াবেটিসের রোগী, আমাকে কি ইনসুলিন নিতেই হবে নাকি ট্যাবলেট খেলেই চলবে?",
    "ডায়াবেটিস হলে কোন ওষুধ কতদিন খেতে হবে?",
    "আমি ডায়াবেটিসের রোগী, আমার HbA1c কত হলে ভালো বলা যাবে?",
    "ডায়াবেটিসের রোগীরা ওষুধ খেলে কি সব ধরনের খাবার খেতে পারবে?",
    "ডায়াবেটিসের ওষুধ কি কিডনি বা হার্টের জন্য ক্ষতিকর?",
    "ওষুধ না খেয়ে শুধু ডায়েট মেনে চললে কি ডায়াবেটিস নিয়ন্ত্রণে রাখা সম্ভব?",
    "ডায়াবেটিসের চিকিৎসায় ভেষজ বা হোমিওপ্যাথিক কি ভালো কাজ করে?",
    "ডায়াবেটিস হলে কী ধরনের ব্যায়াম করা উচিত (যেমন হাঁটা, জগিং ইত্যাদি) এবং দিনে কতক্ষণ ব্যায়াম করা উচিত?",
    "ডায়াবেটিস হলে কি শরীরের ওজন কমানো জরুরি?",
    "ডায়াবেটিস রোগীদের জন্য দিনে কতক্ষণ হাঁটা উচিত?",
    "ডায়াবেটিস রোগীদের জন্য খালি পেটে হাঁটা ভালো নাকি খাওয়ার পর?",
    "ডায়াবেটিস রোগীরা কি স্মোকিং করতে পারবে?",
    "ডায়াবেটিস থেকে কী কী সমস্যা বা জটিলতা হতে পারে?",
    "ডায়াবেটিস হলে পায়ের কেন বেশি যত্ন নিতে হয়?",
    "ডায়াবেটিস রোগীর সুগার কমে গেলে কীভাবে বুঝব এবং কী কী পদক্ষেপ নেব?",
    "ডায়াবেটিস থাকলে কি হার্ট অ্যাটাকের ঝুঁকি বেড়ে যায়?",
    "ডায়াবেটিস রোগীর মাসে অন্তত কতবার সুগার পরীক্ষা করা উচিত?",
    "আমার বারবার প্রস্রাব হচ্ছে এবং বেশি পিপাসা লাগছে, এটা কি ডায়াবেটিসের লক্ষণ?",
    "হঠাৎ করে ওজন কমে যাচ্ছে, এটা কি ডায়াবেটিসের কারণে হতে পারে?",
    "সব সময় ক্লান্ত লাগে, এটা কি ডায়াবেটিসের লক্ষণ?",
    "চোখে ঝাপসা দেখা যাচ্ছে, এটা কি ডায়াবেটিস হওয়ার কারণে হয়েছে?",
    "শরীরে ক্ষত হলে দেরিতে শুকায়, এটা কি ডায়াবেটিসের লক্ষণ?",
    "আমার সুগার লেভেল মাঝে মাঝে বেশি হয়, মাঝে মাঝে স্বাভাবিক থাকে এটা কি ডায়াবেটিসের লক্ষন?",
    "ডায়াবেটিস রোগীর জন্য রমজান মাসে রোজা রেখে ইনসুলিন বা ওষুধ সমন্বয় করার সঠিক নিয়ম কী?",
    "গর্ভাবস্থায় ডায়াবেটিস কেন হয় বা হলে কি তা বাচ্চার ক্ষতি করতে পারে?",
    "আমার বাবা-মায়ের ডায়াবেটিস আছে, আমার হওয়ার সম্ভাবনা কতটুকু?",
    "ডায়াবেটিস হলে কি প্রেগনেন্সিতে কোন সমস্যা হয়?",
    "ডায়াবেটিস কি বংশগত জটিলতা?",
    "ডায়াবেটিস হলে কি আমি স্বাভাবিক জীবন যাপন করতে পারব?",
    "ডায়াবেটিস কি কোনোদিনও ভালো হয় না?",
    "ডায়াবেটিস রোগীর জন্য রমজান মাসে ডায়েট চার্ট-কেমন হবে?",
]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class HybridRetriever:
    """BM25 top-k + dense top-k, BM25 first, de-duplicated by chunk text."""

    def __init__(self, db_dir: str, device: str = "cpu", k: int = TOP_K):
        print(f"Loading {EMBEDDING_MODEL} on {device} ...")
        # Default query instruction of the LangChain BGE wrapper (English):
        # "Represent this question for searching relevant passages: "
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings,
                                collection_name=COLLECTION_NAME)
        collection = self.vector_db.get()
        docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(collection.get("documents") or [], collection.get("metadatas") or [])]
        if not docs:
            raise SystemExit(f"No chunks found in {db_dir}. Run src/Knowledge_Base_Construction.py first.")
        print(f"Loaded {len(docs)} chunks; building the BM25 index ...")
        self.bm25 = BM25Retriever.from_documents(docs)  # default preprocessing: str.split()
        self.bm25.k = k
        self.k = k

    def __call__(self, query: str) -> tuple[list[Document], list[dict]]:
        bm25_docs = self.bm25.invoke(query)
        dense = self.vector_db.similarity_search_with_score(query, k=self.k)
        merged, seen, log = [], set(), []
        for rank, doc in enumerate(bm25_docs, 1):
            log.append(dict(retriever="bm25", rank=rank, chunk_id=doc.metadata.get("chunk_id"),
                            source=doc.metadata.get("source"), distance=None))
        for rank, (doc, dist) in enumerate(dense, 1):
            log.append(dict(retriever="dense", rank=rank, chunk_id=doc.metadata.get("chunk_id"),
                            source=doc.metadata.get("source"), distance=float(dist)))
        for doc in bm25_docs + [d for d, _ in dense]:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)
        return merged, log


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def clean_reasoning_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def source_section(docs: list[Document]) -> str:
    sources = list(dict.fromkeys(doc.metadata.get("source", "Unknown") for doc in docs))
    return "\n\nসূত্র:\n" + "\n".join(f"- [{src}]" for src in sources)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the Qwen3.5-9B Q4_K_M GGUF file")
    ap.add_argument("--db-dir", default="./chroma_bangla_db")
    ap.add_argument("--out", default="results/banglaDiaRAG")
    ap.add_argument("--runs", type=int, default=2, help="independent passes over the 40 questions")
    ap.add_argument("--device", default="cpu", help="device for BGE-M3")
    ap.add_argument("--n-threads", type=int, default=6)
    ap.add_argument("--n-batch", type=int, default=16)
    ap.add_argument("--n-gpu-layers", type=int, default=24,
                help="number of layers offloaded to the GPU (24 in the runs reported in the paper)")
    ap.add_argument("--rope-freq-base", type=float, default=0.0,
                    help="RoPE base passed to llama.cpp; 0 = native value stored in the GGUF file ")
    ap.add_argument("--questions", nargs="*", type=int, help="question numbers to run (default: all 40)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"GGUF file not found: {model_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(args.db_dir, device=args.device)

    print(f"Loading {model_path.name} (threads={args.n_threads}, gpu_layers={args.n_gpu_layers}) ...")
    llm = LlamaCpp(
        model_path=str(model_path),
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_ctx=N_CTX,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        f16_kv=True,
        rope_freq_base=args.rope_freq_base,
        verbose=False,
        stop=STOP,
    )  # seed not set: LangChain default -1 (random)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                            input_variables=["system_instruction", "context", "question"])
    chain = prompt | llm | StrOutputParser() | RunnableLambda(clean_reasoning_tags)

    try:
        import importlib.metadata as md
        versions = {p: md.version(p) for p in ["llama_cpp_python", "langchain-community", "langchain-core",
                                                "langchain-chroma", "chromadb", "rank-bm25",
                                                "sentence-transformers"]}
    except Exception:  # pragma: no cover
        versions = {}
    meta = dict(model_file=model_path.name, temperature=TEMPERATURE, n_ctx=N_CTX, max_tokens=MAX_TOKENS,
                top_k_per_retriever=TOP_K, n_threads=args.n_threads, n_batch=args.n_batch,
                n_gpu_layers=args.n_gpu_layers, rope_freq_base=args.rope_freq_base,
                embedding_model=EMBEDDING_MODEL, collection=COLLECTION_NAME,
                platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                packages=versions)

    selected = args.questions or list(range(1, len(QUESTIONS) + 1))
    all_records = []
    for run in range(1, args.runs + 1):
        started = datetime.now(timezone.utc).isoformat()
        records = []
        print(f"Run {run}/{args.runs}: {len(selected)} questions")
        for qid in selected:
            q = QUESTIONS[qid - 1]
            t0 = time.time()
            print(f"  [{qid}/{len(QUESTIONS)}] ...")
            try:
                chunks, retrieval_log = retriever(q)
                answer = chain.invoke({"context": format_docs(chunks), "question": q,
                                       "system_instruction": SYSTEM_INSTRUCTION})
                final = f"{answer.strip()}{source_section(chunks)}"
                records.append(dict(run=run, id=qid, question=q, answer=final, n_context_chunks=len(chunks),
                                    context_chunk_ids=[c.metadata.get("chunk_id") for c in chunks],
                                    retrieval=retrieval_log, time_sec=round(time.time() - t0, 2),
                                    timestamp_utc=datetime.now(timezone.utc).isoformat()))
            except Exception as exc:  # keep going, but record the failure
                print(f"  Error on Q{qid}: {exc}")
                records.append(dict(run=run, id=qid, question=q, error=str(exc)))
        (out / f"run{run}.json").write_text(
            json.dumps(dict(meta=dict(meta, run=run, started_utc=started), results=records),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        all_records.extend(records)

    with open(out / "responses.jsonl", "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_records)} records to {out}/")


if __name__ == "__main__":
    main()
