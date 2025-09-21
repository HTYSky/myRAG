# main.py (调试版)

import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import time
import os # 引入os模块
import logging
from datetime import datetime

# --- 1. 全局变量与模型加载 ---
app = FastAPI()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str

RAG_ASSETS_PATH = "./assets"
print("Loading chunks and metadata...")
with open(f"{RAG_ASSETS_PATH}/chunks.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)
with open(f"{RAG_ASSETS_PATH}/metadata.json", "r", encoding="utf-8") as f:
    METADATA = json.load(f)

# --- 为FAISS加载过程添加详细的日志埋点 ---
print("\n--- Starting FAISS Index Loading ---")
FAISS_INDEX_FILE = f"{RAG_ASSETS_PATH}/knowledge_base.index"

try:
    # 检查索引文件是否存在
    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"❌ FATAL ERROR: FAISS index file not found at '{FAISS_INDEX_FILE}'")
        exit()
    
    print(f"✅ [Step 1/5] Found FAISS index file at '{FAISS_INDEX_FILE}'.")
    
    # 从硬盘读取索引
    INDEX = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ [Step 2/5] Successfully read index from disk. It contains {INDEX.ntotal} vectors.")

    # 检查GPU环境
    if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
        print("✅ [Step 3/5] GPU detected and faiss-gpu seems installed.")
        
        # 初始化GPU资源
        res = faiss.StandardGpuResources()
        print("✅ [Step 4/5] Successfully initialized GPU resources (StandardGpuResources).")
        
        # 将索引移至GPU
        INDEX = faiss.index_cpu_to_gpu(res, 1, INDEX) # 使用1号GPU
        print("✅ [Step 5/5] Successfully moved FAISS index to GPU-0.")
        
    else:
        print("⚠️ [Step 3/5] GPU not detected or faiss-gpu not installed. Using CPU for FAISS index.")

except Exception as e:
    print(f"\n❌ An error occurred during FAISS index loading:")
    import traceback
    traceback.print_exc()
    exit() # 如果出错则直接退出

print("--- FAISS Index Loading Complete ---\n")
# --- 修改结束 ---


print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer('./model/bge-large-zh-v1.5', device='cuda')

print("Loading reranker model...")
RERANKER_MODEL = CrossEncoder('./model/bge-reranker-large', max_length=512, device='cuda')

print("Loading 4-bit quantized LLM with bitsandbytes...")
LLM_MODEL_PATH = './model/qwen2-7b-instruct-bnb-4bit'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

LLM_MODEL = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)

TEXT_GENERATOR = pipeline(
    "text-generation",
    model=LLM_MODEL,
    tokenizer=TOKENIZER,
    device_map="auto"
)

# ... (后续的 search_and_rerank, build_prompt, 和 API端点代码保持不变) ...
def search_and_rerank(query_text, top_k_retrieve=50, top_k_rerank=3):
    logger.info(f"🔍 开始检索和重排序 - 查询文本: {query_text}")
    
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_embedding = EMBEDDING_MODEL.encode([instruction + query_text], normalize_embeddings=True)
    
    logger.info(f"📊 向量检索 - 检索前 {top_k_retrieve} 个文档")
    distances, indices = INDEX.search(query_embedding.astype('float32'), top_k_retrieve)
    
    if indices.size == 0:
        logger.warning("⚠️ 未找到任何相关文档")
        return []
    
    retrieved_chunks = [CHUNKS[i] for i in indices[0]]
    logger.info(f"📄 检索到 {len(retrieved_chunks)} 个文档片段")
    
    # 记录前5个检索结果的相似度分数
    logger.info("📈 检索结果相似度分数:")
    for i, (dist, idx) in enumerate(zip(distances[0][:5], indices[0][:5])):
        logger.info(f"   {i+1}. 索引{idx}: 距离={dist:.4f}")
    
    pairs = [[query_text, chunk] for chunk in retrieved_chunks]
    scores = RERANKER_MODEL.predict(pairs)
    reranked_results = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
    
    logger.info(f"🎯 重排序完成 - 选择前 {top_k_rerank} 个最相关文档")
    logger.info("📊 重排序分数:")
    for i, (score, chunk) in enumerate(reranked_results[:top_k_rerank]):
        logger.info(f"   {i+1}. 分数={score:.4f}")
    
    final_chunks = [chunk for score, chunk in reranked_results[:top_k_rerank]]
    logger.info(f"✅ 检索和重排序完成，返回 {len(final_chunks)} 个文档片段")
    
    return final_chunks

def build_prompt(question, options, context_chunks):
    logger.info(f"📝 构建提示词 - 问题: {question}")
    logger.info(f"📚 使用 {len(context_chunks)} 个上下文片段")
    
    context = "\n\n".join(context_chunks)
    
    # 记录上下文统计信息和内容
    logger.info(f"📖 上下文统计: 共 {len(context_chunks)} 个片段")
    logger.info("📄 上下文片段内容:")
    for i, chunk in enumerate(context_chunks):
        logger.info(f"   片段 {i+1}: {chunk}")
    
    # 构建选项列表
    options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
    
    prompt = f"""
你是一个严谨的事实核查助手。你的任务是基于提供的【上下文】信息，从给定的选择题中选择最正确的答案。

你必须完全且仅根据【上下文】中的信息来做出判断。严禁使用任何你的先验知识。
这一点至关重要，因为遵循所给文本是评估的严格规则。任何超出上下文的信息都会导致错误的结果。
与此同时，【上下文】和【问题】中也有可能出现大量冗余信息，请务必仔细甄别，不要被冗余信息所误导。

---
【上下文】
{context}
---
【问题】
{question}

【选项】
{options_text}
---

基于你的分析，请从A、B、C、D中选择最正确的答案。你的答案必须是以下四个选项之一：'A'、'B'、'C' 或 'D'。

答案：
"""
    
    logger.info(f"✅ 提示词构建完成，总长度: {len(prompt)} 字符")
    logger.info(f"PROMPT: {prompt}")
    return prompt

@app.post("/answer")
async def get_answer(query: Query):
    start_time = time.time()
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info("=" * 80)
    logger.info(f"🚀 新请求开始 - 请求ID: {request_id}")
    logger.info(f"📝 接收到的输入数据:")
    logger.info(f"   问题: {query.question}")
    logger.info(f"   选项A: {query.option_A}")
    logger.info(f"   选项B: {query.option_B}")
    logger.info(f"   选项C: {query.option_C}")
    logger.info(f"   选项D: {query.option_D}")
    
    question = query.question
    options = {
        "A": query.option_A,
        "B": query.option_B,
        "C": query.option_C,
        "D": query.option_D,
    }
    
    logger.info("🔄 开始处理整个问题")
    
    # 构建完整的问题文本用于检索
    full_question = f"{question} {' '.join(options.values())}"
    logger.info(f"📋 完整问题文本: {full_question}")
    
    # 检索和重排序
    context_chunks = search_and_rerank(full_question)
    
    if not context_chunks:
        logger.warning("⚠️ 未找到相关上下文，返回默认答案A")
        final_answer = "A"
    else:
        # 构建提示词
        prompt = build_prompt(question, options, context_chunks)
        
        # LLM 推理
        logger.info("🤖 开始LLM推理")
        llm_start_time = time.time()
        
        outputs = TEXT_GENERATOR(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0
        )
        
        llm_end_time = time.time()
        generated_text = outputs[0]['generated_text']
        answer = generated_text[len(prompt):].strip()
        
        logger.info(f"⏱️ LLM推理耗时: {llm_end_time - llm_start_time:.2f}秒")
        logger.info(f"🎯 LLM原始输出: '{answer}'")
        
        # 解析答案
        final_answer = "A"  # 默认答案
        for option_key in ["A", "B", "C", "D"]:
            if option_key in answer.upper():
                final_answer = option_key
                break
        
        logger.info(f"🏆 最终答案: {final_answer}")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"⏱️ 总处理时间: {total_time:.2f}秒")
    logger.info(f"✅ 请求完成 - 请求ID: {request_id}")
    logger.info("=" * 80)
    
    return {"correct_answer": final_answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)