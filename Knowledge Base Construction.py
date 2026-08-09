import os
import re
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------
# TASK 1.3: Custom Semantic Chunking & Metadata Extraction
# ---------------------------------------------------------
def process_markdown_guideline(file_path, source_prefix):
    print(f"📖 Reading and analyzing Markdown file: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found in current directory.")
        return []

    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Windows \r\n ফিক্স করা
    markdown_text = markdown_text.replace('\r\n', '\n')

    # 🚀 ULTIMATE FIX: নিউলাইনের ওপর নির্ভর না করে সরাসরি হেডার (**) দিয়ে ভাগ করা!
    # এটি ঠিক সেই জায়গাগুলোতে ফাইলকে কাটবে যেখানে নতুন লাইনের শুরুতে ** আছে।
    raw_blocks = re.split(r'(?m)^(?=\*\*.*?\*\*)', markdown_text.strip())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=100,
        separators=["\n\n\n\n", "\n\n", "\n", "।", " ", ""]
    )

    final_documents = []
    
    for idx, block in enumerate(raw_blocks):
        block = block.strip()
        if not block:
            continue
            
        header_match = re.match(r'\*\*(.*?)\*\*', block)
        if header_match:
            header_name = header_match.group(1).strip()
            raw_content = block[header_match.end():].strip()
        else:
            header_name = "General Topic"
            raw_content = block
            
        source_match = re.search(r'\+\+\+\s*\\?\[(.*?)\]', raw_content)
        
        if source_match:
            raw_source = source_match.group(1).strip()
            if raw_source.lower().startswith("source:"):
                source_name = raw_source[7:].strip()
            else:
                source_name = raw_source
                
            cleaned_content = re.sub(r'\+\+\+\s*\\?\[.*?\]', '', raw_content).strip()
        else:
            source_name = f"{source_prefix} General"
            cleaned_content = raw_content

        sub_chunks = text_splitter.split_text(cleaned_content)
        
        for sub_idx, sub_chunk in enumerate(sub_chunks):
            chunk_id = f"{source_prefix}_P{idx+1}_C{sub_idx+1}_BN"
            
            doc = Document(
                page_content=f"বিষয়: {header_name}\nতথ্য: {sub_chunk.strip()}",
                metadata={
                    "chunk_id": chunk_id, 
                    "source": source_name, 
                    "topic": header_name,
                    "category": source_prefix 
                }
            )
            final_documents.append(doc)
            
    print(f"✅ Created {len(final_documents)} semantic chunks with metadata.")
    return final_documents

# ---------------------------------------------------------
# TASK 1.4: Vector Embedding and Database Storage (CPU ONLY)
# ---------------------------------------------------------
def store_in_vector_db(documents, db_directory="./chroma_bangla_db"):
    # Force CPU to avoid GTX driver/DLL conflicts
    device = "cpu"
    print(f"\n🔄 Loading BAAI/bge-m3 embedding model on [{device}]...")
    
    # BGE-M3 is heavy; it will use your Ryzen cores automatically
    model_kwargs = {'device': device} 
    encode_kwargs = {'normalize_embeddings': True}
    
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"🧬 Generating embeddings for {len(documents)} chunks...")
    print("Note: This may take 2-5 minutes on CPU depending on file size.")
    
    # Persisting to local disk
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=bge_embeddings,
        persist_directory=db_directory,
        collection_name="bangla_diabetes_guidelines"
    )
    
    print(f"📂 Knowledge base successfully stored in: {db_directory}")
    return vector_store

# ---------------------------------------------------------
# EXECUTION PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure your file is named exactly this or change it here
    markdown_file_path = "bangla_guidelines_new.md" 
    
    # Start the process
    docs = process_markdown_guideline(markdown_file_path, "DIAB_GUIDE_BN")
    
    if docs:
        vector_db = store_in_vector_db(docs)
        print("\n🎉 Phase 1 Complete!")
        print("The 'chroma_bangla_db' folder has been created.")
        print("You can now run your RAG script (Code 1).")
    else:
        print("⚠️ No documents were processed. Please check your .md file path.")