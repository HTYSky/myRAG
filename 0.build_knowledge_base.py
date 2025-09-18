import os
import json
import fitz  # PyMuPDF
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from tqdm import tqdm
import numpy as np

# PDF文档存放路径
PDF_PATH = "/dataset/KnowledgeDocument/"
# 处理后数据和索引的保存路径
OUTPUT_PATH = "./assets"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def parse_pdf_hybrid(file_path):
    """
    使用PyMuPDF和pdfplumber混合策略解析PDF。
    PyMuPDF用于快速提取文本，pdfplumber用于精准提取表格。
    """
    full_text = ""
    try:
        # 使用PyMuPDF快速提取文本
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                full_text += page.get_text("text") + "\n\n"

        # 使用pdfplumber提取表格并转换为Markdown格式
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # 将列表的列表转换为Markdown表格字符串
                        markdown_table = "| " + " | ".join(str(header) for header in table) + " |\n"
                        markdown_table += "| " + " | ".join(["---"] * len(table)) + " |\n"
                        for row in table[1:]:
                            markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                        full_text += f"\n--- TABLE ---\n{markdown_table}\n--- END TABLE ---\n\n"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return full_text


def chunk_text(documents):
    """
    对提取的文档内容进行分块。
    """
    # 首先，可以考虑使用Markdown标题进行宏观分块（如果适用）
    # 这里为简化，直接使用递归字符分割，它本身会优先考虑段落和句子
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # 每个块的最大长度
        chunk_overlap=100, # 块之间的重叠
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "], # 优先使用的分隔符
    )
    
    all_chunks = []
    doc_metadata = [] # 存储每个chunk的来源文档

    for doc_name, content in tqdm(documents.items(), desc="Chunking documents"):
        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
        doc_metadata.extend([doc_name] * len(chunks))
        
    return all_chunks, doc_metadata


def main():
    # 1. 解析所有PDF文档
    all_documents = {}
    
    # 检查PDF路径是否存在
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF directory not found at {PDF_PATH}")
        return

    # 使用 os.listdir() 获取所有文件名
    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {PDF_PATH}")
        return
        
    # --- 只处理前5个文件，用于测试 ---
    pdf_files_subset = pdf_files[:5]
    print(f"--- Running in test mode: Processing the first {len(pdf_files_subset)} PDF files. ---")
    # ------------------------------------

    # 使用切片后的子集进行处理
    for filename in tqdm(pdf_files_subset, desc="Parsing PDFs"):
        file_path = os.path.join(PDF_PATH, filename)
        content = parse_pdf_hybrid(file_path)
        if content:
            all_documents[filename] = content

    # 2. 文本分块
    chunks, metadata = chunk_text(all_documents)
    
    if not chunks:
        print("Error: No text chunks were generated after parsing. Cannot proceed.")
        return

    # 保存chunks和metadata，以便API服务加载
    with open(os.path.join(OUTPUT_PATH, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 3. 加载嵌入模型
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('./model/bge-large-zh-v1.5', device='cuda')

    # 4. 生成嵌入向量
    instruction = "为这个句子生成表示以用于检索相关文章："
    chunks_with_instruction = [instruction + chunk for chunk in chunks]
    
    print("Generating embeddings...")
    embeddings = embedding_model.encode(chunks_with_instruction, 
                                        batch_size=32, 
                                        show_progress_bar=True, 
                                        normalize_embeddings=True)

    # 5. 构建并保存FAISS索引
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, os.path.join(OUTPUT_PATH, "knowledge_base.index"))
    print(f"Knowledge base built successfully! Index and data saved in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()