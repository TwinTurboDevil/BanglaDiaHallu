import os
import re
import json
import torch
import time
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# =========================================================
# 1. HARDWARE & PATH CONFIGURATION (Optimized for CPU)
# =========================================================
DEVICE = "cpu" 
DB_DIR = "./chroma_bangla_db"
LOCAL_MODEL_DIR = "./models"
FILENAME = "Add the gguf file" 

MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, FILENAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Error: আপনার ম্যানুয়ালি ডাউনলোড করা মডেল ফাইলটি এই পাথে খুঁজে পাওয়া যায়নি: '{MODEL_PATH}'\n"
        f"অনুগ্রহ করে নিশ্চিত করুন যে ফাইলটি সঠিকভাবে 'models' ফোল্ডারের ভেতর রাখা হয়েছে।"
    )
else:
    print(f"✅ Manually downloaded model found at: {MODEL_PATH}")

# =========================================================
# 2. HYBRID RETRIEVER INITIALIZATION
# =========================================================
print(f"🔄 Loading BGE-M3 Embeddings on {DEVICE}...")
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

print("📚 Reconstructing Vector Database and BM25 Index...")
vector_db = Chroma(
    persist_directory=DB_DIR, 
    embedding_function=bge_embeddings,
    collection_name="bangla_diabetes_guidelines"
)

collection = vector_db.get()
docs_for_bm25 = []
if collection and 'documents' in collection:
    for i in range(len(collection['documents'])):
        doc_text = collection['documents'][i]
        doc_meta = collection['metadatas'][i] if collection['metadatas'] else {}
        docs_for_bm25.append(Document(page_content=doc_text, metadata=doc_meta))

if not docs_for_bm25:
    print("⚠️ Warning: No documents found in your vector collection to build BM25 index.")
    docs_for_bm25 = [Document(page_content="Placeholder text for empty DB initialization", metadata={"source": "None"})]

bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
bm25_retriever.k = 3 

vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

def hybrid_retriever_func(query):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    all_docs = bm25_docs + vector_docs
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    
    return unique_docs

# =======================
# 3. LLM CONFIGURATION 
# =======================
print(f"🧊 Initializing Qwen 3.5 on CPU (Threads: 6)...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=0,      
    n_threads=6,         
    n_batch=16,          
    n_ctx=8192,          
    temperature=0.5,     
    max_tokens=8192,      
    f16_kv=True,
    verbose=False,
    stop=["<|im_end|>", "<|endoftext|>"] 
)

# 🚀 NEW: আপনার দেওয়া উদাহরণ অনুযায়ী মাইক্রো-টেমপ্লেট (Micro-Template) যুক্ত করা হয়েছে
SYSTEM_INSTRUCTION = """
### ROLE & TONE
You are a Senior Medical Consultant for diabetes care in Bangladesh. 
Tone: Empathetic, highly professional, encouraging.

### OPERATIONAL RULES
1. LANGUAGE: Code-mixed Bangla. Write in Bangla script but keep key medical/dietary terms in English inside brackets (e.g., ব্লাড সুগার (Blood Sugar), ফাইবার (Fiber)).
2. GROUNDING: Answer strictly based on the CONTEXT provided. If context is missing, say: "এই প্রশ্নের উত্তরের জন্য দয়া করে আপনার ডাক্তারের পরামর্শ নিন।"
3. NO REFERENCES IN TEXT: Do not write source names in your response. 

### RESPONSE STRUCTURE (MUST FOLLOW)
1. Start with an empathetic opening (e.g., "আপনার প্রশ্নটি খুবই গুরুত্বপূর্ণ...").
2. Use bullet points or numbered lists (১, ২) for explaining items or giving rules.
3. Include a clear "আমার পরামর্শ হলো:" section with actionable steps.
4. End with a clinical disclaimer ("একটি জরুরি কথা: প্রত্যেকের শরীর আলাদা...") and a warm closing ("নিজের যত্ন নিন!").
"""

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

prompt_template = PromptTemplate(
    template="""<|im_start|>system
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
""",
    input_variables=["system_instruction", "context", "question"]
)

def clean_reasoning_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

generation_chain = prompt_template | llm | StrOutputParser() | RunnableLambda(clean_reasoning_tags)

# =========================================================
# 4. EXPERIMENT EXECUTION (40 QUESTIONS)
# =========================================================
questions = [
    "ডায়াবেটিস হলে কি আমি দিনে তিন বেলা ভাত খেতে পারব নাকি ভাতের বদলে রুটি খাওয়া যাবে?",
    "ডায়াবেটিস হলে কি আমার জন্য মিষ্টি ফল (যেমন: আম, কাঁঠাল) খাওয়া একদমই নিষেধ, নাকি নির্দিষ্ট পরিমাণে খাওয়া যাবে?",
    "ডায়াবেটিস হলে কি আমি চিনি ছাড়া চা খেতে পারবো আর খেতে পারলে দিনে কত কাপ খেতে পারবো?",
    "ডায়াবেটিস রোগীর জন্য কি মধু বা গুড় চিনির বিকল্প হিসেবে ব্যবহার করা নিরাপদ?",
    "ডায়াবেটিস হলে কি আমি মিষ্টি খেতে পারবো?",
    "ডায়াবেটিস রোগীর জন্য কি কি শাক-সবজি খাওয়া ভালো?",
    "ডায়াবেতিসের রোগীরা ইফতারে খেজুর বা শরবত খেতে পারবে কি?",
    "ডায়াবেটিস হলে আমি কি কি খাবো আর কি কি এড়িয়ে চলবো?",
    "ডায়াবেটিস রোগীর জন্য ডায়াবেটিস ডায়েট-চার্ট কেমন হওয়া উচিত?",
    "আমি ডায়াবেতিসের রোগী, আমাকে কি ইনসুলিন নিতেই হবে নাকি ট্যাবলেট খেলেই চলবে?",
    "ডায়াবেটিস হলে কোন ওষুধ কতদিন খেতে হবে?",
    "আমি ডায়াবেতিসের রোগী, আমার HbA1c কত হলে ভালো বলা যাবে?",
    "ডায়াবেতিসের রোগীরা ওষুধ খেলে কি সব ধরনের খাবার খেতে পারবে?",
    "ডায়াবেতিসের ওষুধ কি কিডনি বা হার্টের জন্য ক্ষতিকর?",
    "ওষুধ না খেয়ে শুধু ডায়েট মেনে চললে কি ডায়াবেটিস নিয়ন্ত্রণে রাখা সম্ভব?",
    "ডায়াবেতিসের চিকিৎসায় ভেষজ বা হোমিওপ্যাথিক কি ভালো কাজ করে?",
    "ডায়াবেটিস হলে কী ধরনের ব্যায়াম করা উচিত (যেমন হাঁটা, জগিং ইত্যাদি) এবং দিনে কতক্ষণ ব্যায়াম করা উচিত?",
    "ডায়াবেটিস হলে কি আমি শরীরের ওজন কমানো জরুরি?",
    "ডায়াবেটিস রোগীদের জন্য দিনে কতক্ষণ হাঁটা উচিত?",
    "ডায়াবেটিস রোগীদের জন্য খালি পেটে হাঁটা ভালো নাকি খাওয়ার পর?",
    "ডায়াবেটিস রোগীরা কি স্মোকিং করতে পারবে?",
    "ডায়াবেটিস থেকে কী কী সমস্যা বা জটিলতা হতে পারে?",
    "ডায়াবেটিস হলে পায়ের কেন বিশেষ যত্ন নিতে হয়?",
    "ডায়াবেটিস রোগীর সুগার কমে গেলে কীভাবে বুঝব এবং কী কী পদক্ষেপ নেব?",
    "ডায়াবেটিস থাকলে কি হার্ট অ্যাটাকের ঝুঁকি বেড়ে যায়?",
    "ডায়াবেটিস রোগীর মাসে অন্তত কতবার সুগার পরীক্ষা করা উচিত?",
    "আমার বারবার প্রস্রাব হচ্ছে এবং খুব বেশি পিপাসা লাগছে, এটা কি ডায়াবেতিসের লক্ষণ?",
    "হঠাৎ করে ওজন কমে যায়, এটা কি ডায়াবেতিসের কারণে হতে পারে?",
    "সব সময় ক্লান্ত লাগে, এটা কি ডায়াবেতিসের লক্ষণ?",
    "চোখে ঝাপসা দেখা যাচ্ছে, এটা কি ডায়াবেটিস হওয়ার কারণে হয়েছে?",
    "শরীরে ক্ষত হলে দেরিতে শুকায়, এটা কি ডায়াবেতিসের লক্ষণ?",
    "আমার সুগার লেভেল মাঝে মাঝে বেশি হয়, মাঝে মাঝে স্বাভাবিক থাকে এটা কি ডায়াবেতিসের লক্ষন?",
    "ডায়াবেটিস রোগীর জন্য রমজান মাসে রোজা রেখে ইনসুলিন বা ওষুধ সমন্বয় করার সঠিক নিয়ম কী?",
    "গর্ভাবস্থায় ডায়াবেটিস কেন হয় বা হলে কি তা বাচ্চার ক্ষতি করতে পারে?",
    "আমার বাবা-মায়ের ডায়াবেটিস আছে, আমার হওয়ার সম্ভাবনা কতটুকু?",
    "ডায়াবেটিস হলে কি প্রেগনেন্সিতে কোন সমস্যা হয়?",
    "ডায়াবেটিস কি বংশগত জটিলতা?",
    "ডায়াবেটিস হলে কি আমি স্বাভাবিক জীবন যাপন করতে পারব?",
    "ডায়াবেটিস কি কোনোদিনও ভালো হয় না?",
    "ডায়াবেটিস রোগীর জন্য রমজান মাসে ডায়েট চার্ট কেমন হবে?"
]

results = []
start_total = time.time()

print(f"🎬 Starting Experiment: {len(questions)} Questions.")

for i, q in enumerate(questions):
    iter_start = time.time()
    print(f"[{i+1}/{len(questions)}] Processing...")
    
    try:
        retrieved_chunks = hybrid_retriever_func(q)
        context_string = format_docs(retrieved_chunks)
        
        llm_answer = generation_chain.invoke({
            "context": context_string,
            "question": q,
            "system_instruction": SYSTEM_INSTRUCTION
        })
        
        unique_sources = set()
        for doc in retrieved_chunks:
            source_name = doc.metadata.get('source', 'Unknown')
            unique_sources.add(source_name)
            
        source_section = "\n\nসূত্র:\n" + "\n".join([f"- [{src}]" for src in unique_sources])
        final_answer_with_sources = f"{llm_answer.strip()}{source_section}"
        
        results.append({
            "id": i+1,
            "question": q,
            "answer": final_answer_with_sources,
            "time_sec": round(time.time() - iter_start, 2)
        })
        
    except Exception as e:
        print(f"⚠️ Error on Q{i+1}: {e}")

# Save output data safely
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

total_time = round((time.time() - start_total) / 60, 2)
print(f"✅ Complete in {total_time} minutes.")