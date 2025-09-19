import os
import json
import fitz  # PyMuPDF
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from tqdm import tqdm
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings
import sys
from io import StringIO

# PDF文档存放路径
PDF_PATH = "/dataset/KnowledgeDocument/"
# 处理后数据和索引的保存路径
OUTPUT_PATH = "./assets"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 状态文件路径
STATE_FILE = os.path.join(OUTPUT_PATH, "processing_state.pkl")
LOG_FILE = os.path.join(OUTPUT_PATH, "processing.log")

class ProcessingState:
    """处理状态管理类"""
    def __init__(self):
        self.processed_files = set()  # 已成功处理的文件
        self.failed_files = set()     # 处理失败的文件
        self.start_time = None
        self.last_update = None
        self.total_files = 0
        self.current_chunks = []
        self.current_metadata = []
        
    def save(self):
        """保存状态到文件"""
        self.last_update = datetime.now()
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls):
        """从文件加载状态"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        return cls()
    
    def get_progress(self):
        """获取处理进度"""
        total = self.total_files
        processed = len(self.processed_files)
        failed = len(self.failed_files)
        remaining = total - processed - failed
        return processed, failed, remaining, total

def suppress_pdf_warnings():
    """抑制PDF处理过程中的无关紧要警告"""
    # 抑制PyMuPDF的颜色相关警告
    warnings.filterwarnings("ignore", message=".*Cannot set gray.*color.*")
    warnings.filterwarnings("ignore", message=".*invalid float value.*")
    
    # 抑制pdfplumber的警告
    warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
    
    # 抑制其他PDF相关的警告
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="fitz")
    warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_pdf_hybrid(file_path, logger):
    """
    使用PyMuPDF和pdfplumber混合策略解析PDF。
    PyMuPDF用于快速提取文本，pdfplumber用于精准提取表格。
    """
    full_text = ""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
            
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"文件为空: {file_path}")
            return None
            
        logger.info(f"开始处理PDF: {os.path.basename(file_path)} (大小: {file_size/1024/1024:.2f}MB)")
        
        # 临时抑制警告
        with warnings.catch_warnings():
            suppress_pdf_warnings()
            
            # 使用PyMuPDF快速提取文本
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                logger.info(f"PDF页数: {page_count}")
                
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text("text")
                        if page_text.strip():  # 只添加非空页面
                            full_text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"页面 {page_num+1} 处理失败: {e}")

            # 使用pdfplumber提取表格并转换为Markdown格式
            try:
                with pdfplumber.open(file_path) as pdf:
                    table_count = 0
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            tables = page.extract_tables()
                            for table in tables:
                                if table and len(table) > 1:  # 确保表格有数据
                                    # 将列表的列表转换为Markdown表格字符串
                                    markdown_table = "| " + " | ".join(str(header) for header in table[0]) + " |\n"
                                    markdown_table += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
                                    for row in table[1:]:
                                        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                                    full_text += f"\n--- TABLE {table_count+1} ---\n{markdown_table}\n--- END TABLE ---\n\n"
                                    table_count += 1
                        except Exception as e:
                            logger.warning(f"页面 {page_num+1} 表格提取失败: {e}")
                            
                    logger.info(f"提取到 {table_count} 个表格")
                    
            except Exception as e:
                logger.warning(f"pdfplumber处理失败，仅使用PyMuPDF结果: {e}")
            
        if not full_text.strip():
            logger.warning(f"PDF文件无有效文本内容: {file_path}")
            return None
            
        logger.info(f"PDF处理完成: {os.path.basename(file_path)}, 文本长度: {len(full_text)} 字符")
        return full_text
        
    except Exception as e:
        logger.error(f"PDF处理失败 {file_path}: {e}")
        return None


def chunk_text(documents, logger):
    """
    对提取的文档内容进行分块。
    """
    logger.info("开始文本分块处理...")
    
    # 首先，可以考虑使用Markdown标题进行宏观分块（如果适用）
    # 这里为简化，直接使用递归字符分割，它本身会优先考虑段落和句子
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # 每个块的最大长度
        chunk_overlap=100, # 块之间的重叠
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "], # 优先使用的分隔符
    )
    
    all_chunks = []
    doc_metadata = [] # 存储每个chunk的来源文档

    for doc_name, content in tqdm(documents.items(), desc="分块处理文档"):
        try:
            chunks = text_splitter.split_text(content)
            all_chunks.extend(chunks)
            doc_metadata.extend([doc_name] * len(chunks))
            logger.info(f"文档 {doc_name} 分块完成，生成 {len(chunks)} 个块")
        except Exception as e:
            logger.error(f"文档 {doc_name} 分块失败: {e}")
            
    logger.info(f"文本分块完成，总共生成 {len(all_chunks)} 个块")
    return all_chunks, doc_metadata


def estimate_remaining_time(processed, total, start_time):
    """估算剩余时间"""
    if processed == 0:
        return "未知"
    
    elapsed = time.time() - start_time
    avg_time_per_file = elapsed / processed
    remaining_files = total - processed
    estimated_remaining = avg_time_per_file * remaining_files
    
    return str(timedelta(seconds=int(estimated_remaining)))

def process_pdfs_with_progress(pdf_files, state, logger):
    """处理PDF文件，带进度条和时间估算"""
    all_documents = {}
    
    # 过滤出未处理的文件
    remaining_files = [f for f in pdf_files if f not in state.processed_files and f not in state.failed_files]
    
    if not remaining_files:
        logger.info("所有PDF文件已处理完成")
        return all_documents
    
    logger.info(f"开始处理 {len(remaining_files)} 个PDF文件")
    
    # 创建进度条
    pbar = tqdm(remaining_files, desc="处理PDF文件", unit="文件")
    
    for filename in pbar:
        file_path = os.path.join(PDF_PATH, filename)
        
        try:
            content = parse_pdf_hybrid(file_path, logger)
            if content:
                all_documents[filename] = content
                state.processed_files.add(filename)
                logger.info(f"✓ 成功处理: {filename}")
            else:
                state.failed_files.add(filename)
                logger.error(f"✗ 处理失败: {filename}")
                
        except Exception as e:
            state.failed_files.add(filename)
            logger.error(f"✗ 处理异常 {filename}: {e}")
        
        # 更新进度信息
        processed, failed, remaining, total = state.get_progress()
        pbar.set_postfix({
            '已处理': processed,
            '失败': failed,
            '剩余': remaining,
            '剩余时间': estimate_remaining_time(processed, total, state.start_time)
        })
        
        # 定期保存状态
        if processed % 10 == 0:  # 每处理10个文件保存一次
            state.save()
            logger.info(f"状态已保存: 已处理 {processed}/{total} 文件")
    
    pbar.close()
    return all_documents

def main():
    # 抑制PDF处理过程中的无关紧要警告
    suppress_pdf_warnings()
    
    # 设置日志
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("开始构建知识库")
    logger.info("=" * 60)
    
    # 加载或创建处理状态
    state = ProcessingState.load()
    
    # 检查PDF路径是否存在
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF目录不存在: {PDF_PATH}")
        return

    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"在 {PDF_PATH} 中未找到PDF文件")
        return
    
    # 初始化状态
    if state.start_time is None:
        state.start_time = time.time()
        state.total_files = len(pdf_files)
        logger.info(f"发现 {len(pdf_files)} 个PDF文件")
    else:
        logger.info(f"恢复处理: 发现 {len(pdf_files)} 个PDF文件")
    
    # 显示当前状态
    processed, failed, remaining, total = state.get_progress()
    logger.info(f"处理状态: 已处理 {processed}, 失败 {failed}, 剩余 {remaining}, 总计 {total}")
    
    # 1. 处理PDF文档
    logger.info("开始PDF文档处理阶段...")
    all_documents = process_pdfs_with_progress(pdf_files, state, logger)
    
    if not all_documents:
        logger.error("没有成功处理任何PDF文件，无法继续")
        return
    
    # 保存状态
    state.save()
    
    # 2. 文本分块
    logger.info("开始文本分块阶段...")
    chunks, metadata = chunk_text(all_documents, logger)
    
    if not chunks:
        logger.error("文本分块失败，无法继续")
        return

    # 保存chunks和metadata
    logger.info("保存分块数据...")
    with open(os.path.join(OUTPUT_PATH, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"分块数据已保存: {len(chunks)} 个块")

    # 3. 加载嵌入模型
    logger.info("加载嵌入模型...")
    try:
        embedding_model = SentenceTransformer('./model/bge-large-zh-v1.5', device='cuda')
        logger.info("嵌入模型加载成功")
    except Exception as e:
        logger.error(f"嵌入模型加载失败: {e}")
        return

    # 4. 生成嵌入向量
    logger.info("开始生成嵌入向量...")
    instruction = "为这个句子生成表示以用于检索相关文章："
    chunks_with_instruction = [instruction + chunk for chunk in chunks]
    
    try:
        embeddings = embedding_model.encode(chunks_with_instruction, 
                                            batch_size=32, 
                                            show_progress_bar=True, 
                                            normalize_embeddings=True)
        logger.info(f"嵌入向量生成完成: {embeddings.shape}")
    except Exception as e:
        logger.error(f"嵌入向量生成失败: {e}")
        return

    # 5. 构建并保存FAISS索引
    logger.info("构建FAISS索引...")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        faiss.write_index(index, os.path.join(OUTPUT_PATH, "knowledge_base.index"))
        logger.info("FAISS索引构建完成")
    except Exception as e:
        logger.error(f"FAISS索引构建失败: {e}")
        return
    
    # 清理状态文件
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logger.info("处理状态文件已清理")
    
    # 计算总耗时
    total_time = time.time() - state.start_time
    logger.info("=" * 60)
    logger.info(f"知识库构建完成!")
    logger.info(f"总耗时: {timedelta(seconds=int(total_time))}")
    logger.info(f"处理文件: {len(state.processed_files)} 成功, {len(state.failed_files)} 失败")
    logger.info(f"生成块数: {len(chunks)}")
    logger.info(f"索引维度: {dimension}")
    logger.info(f"数据保存路径: {OUTPUT_PATH}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()