import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 配置路径 (与之前保持一致) ---
model_path = './model/Qwen2-7B-Instruct'
# 保存路径建议用新的名字，以明确是bitsandbytes量化版
quant_path = './model/qwen2-7b-instruct-bnb-4bit'

def main():
    print(f"Loading model from local path: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist!")
        return

    try:
        # --- 核心修改：使用bitsandbytes进行加载时量化 ---
        
        # 1. 定义bitsandbytes的4-bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. 在加载模型时直接应用量化配置
        # 这会在模型加载到GPU时，自动应用4-bit量化
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto", # bitsandbytes需要device_map
            trust_remote_code=True
        )
        
        # 3. 加载分词器 (这部分不变)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 4. 保存已经量化好的模型
        # 加载完成后，模型在内存中已经是4-bit的了
        print(f"Saving 4-bit quantized model to: {quant_path}")
        model.save_pretrained(quant_path)
        tokenizer.save_pretrained(quant_path)

        print("4-bit model and tokenizer saved successfully!")
        print("You can now load this model from the new path for inference.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()