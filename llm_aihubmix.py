import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import csv
import sys
from dotenv import load_dotenv


# ========== 配置日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_call.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
# 定义一个正则表达式，匹配 Excel (XML) 不允许的控制字符
# (ASCII 0-8, 11-12, 14-31)
illegal_chars_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

# ========== 配置 ==========
temperature = float(sys.argv[1])
file_use = "AllQuestions.xlsx"

API_KEY = os.getenv("AIHUBMIX_API_KEY")
if not API_KEY:
    logger.critical("环境变量 AIHUBMIX_API_KEY 未设置！请在.env文件中配置")
    sys.exit(1)  # 退出程序

API_URL = "https://api.aihubmix.com/v1/chat/completions"
MODELS = [
    ### Remember: keep the cheapest models at the front to save costs!
    ### OpenAI Models
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "o4-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-5",

    ### Qwen Models
    "qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-next-80b-a3b-instruct",
    "qwen3-next-80b-a3b-thinking",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-3b-instruct",

    ### llama Models
    "llama-3.3-70b",
    "llama3.1-8b",

    ### Google gemini Models
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    
    ### Doubao Models
    "doubao-seed-1-6",
    "doubao-seed-1-6-flash",
    "doubao-seed-1-6-thinking",
    
    ### Deepseek Models
    "DeepSeek-R1",
    "DeepSeek-V3",
    "DeepSeek-V3.1-Terminus",
    "deepseek-v3.2",
    "DeepSeek-V3.1-Fast",
    
    # deepseek-ai
    "deepseek-ai/DeepSeek-V2.5",

    # 美团
    "LongCat-Flash-Chat",

    # inclusionAI
    "inclusionAI/Ling-flash-2.0",
    "inclusionAI/Ring-flash-2.0",
    "inclusionAI/Ling-mini-2.0",

    # ernie
    "ernie-5.0-thinking-preview",
    "ernie-4.5-turbo-latest",
    "ernie-4.5",
    "ernie-x1-turbo-32k-preview",
    "ernie-x1.1-preview"

]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ========== 全局 Session 复用 ==========
session = requests.Session()
retries = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# ========== 模型调用 ==========
def query_model(api_key, model, prompt, question, max_retries=2):
    messages = [{"role": "user", "content": f"{prompt} {question}"}]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    models_need_max_completion = ["gpt-5", "gpt-5-mini", "o3", "o4-mini"]
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens" if model in models_need_max_completion else "max_tokens": 1200
    }

    backoff = 2
    for retry in range(max_retries + 1):
        try:
            start = time.perf_counter()
            response = session.post(API_URL, headers=headers, json=data, timeout=(10, 120))
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            elapsed = time.perf_counter() - start
            logger.info(f"✅ {model} 成功，耗时 {elapsed:.2f}s")
            return model, content
        except Exception as e:
            if retry < max_retries:
                logger.warning(f" {model} 失败({e})，{backoff}s 后重试...")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(f" {model} 最终失败: {e}")
                return model, f"模型 {model} 调用失败: {str(e)}"


# ========== Excel处理（分批版） ==========
def process_row(index, row, df, sheet_name):
    """处理单行数据，调用所有模型"""
    prompt = row.get('Prompt', '')
    prompt_cn = row.get('Prompt_CN', '')
    question = row.get('Question', '')
    question_cn = row.get('Question_CN', '')
    
    logger.info(f"Processing row {index + 1} in sheet {sheet_name}")

    for model in MODELS:
        # 英文调用（仅当单元格为空时）
        if pd.isna(df.at[index, model]) or str(df.at[index, model]).strip() == "":
            _, response_eng = query_model(API_KEY, model, prompt, question)
            df.at[index, model] = response_eng
            time.sleep(0.5)  # 单模型调用间隔（可调整）
        
        # 中文调用（仅当单元格为空时）
        if pd.isna(df.at[index, f"{model}_CN"]) or str(df.at[index, f"{model}_CN"]).strip() == "":
            _, response_cn = query_model(API_KEY, model, prompt_cn, question_cn)
            df.at[index, f"{model}_CN"] = response_cn
            time.sleep(0.5)  # 单模型调用间隔（可调整）


def process_batch(batch_rows, df, sheet_name, batch_num, total_batches):
    """处理一批行数据"""
    logger.info(f"开始处理 {sheet_name} 的第 {batch_num}/{total_batches} 批，共 {len(batch_rows)} 行")
    
    with ThreadPoolExecutor(max_workers=5) as executor:  # 每批内5线程并发
        futures = [
            executor.submit(process_row, idx, row, df, sheet_name)
            for idx, row in batch_rows
        ]
        
        # 等待批次内所有行处理完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"批次 {batch_num} 处理行时出错: {str(e)}")
    
    logger.info(f"{sheet_name} 的第 {batch_num}/{total_batches} 批处理完成")


def process_excel_file(file_path, batch_size=20):
    """分批处理Excel文件，支持多Sheet"""
    excel_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    for sheet_name, df in excel_sheets.items():
        total_rows = len(df)
        logger.info(f"开始处理表格 {sheet_name}，共 {total_rows} 行，每批处理 {batch_size} 行")

        output_path = OUTPUT_DIR / f"{file_path.stem}_aihub_temp{temperature}.xlsx"

        # 为每个模型添加结果列（中英文）
        for model in MODELS:
            eng_col = model
            cn_col = f"{model}_CN"
            if eng_col not in df.columns:
                df[eng_col] = ""
            if cn_col not in df.columns:
                df[cn_col] = ""

        # 分割批次（将行索引和数据打包成列表）
        rows = list(df.iterrows())  # 格式: [(index0, row0), (index1, row1), ...]
        total_batches = (total_rows + batch_size - 1) // batch_size  # 向上取整计算总批数
        batches = [rows[i:i + batch_size] for i in range(0, total_rows, batch_size)]

        # 逐批处理
        for batch_num, batch in enumerate(batches, 1):
            process_batch(batch, df, sheet_name, batch_num, total_batches)
            
            for col in df.select_dtypes(include=['object']).columns:
                str_col = df[col].astype(str)
                mask = str_col.str.contains(illegal_chars_re, regex=True, na=False)
                df[col] = df[col].mask(mask, "")

            # 每批处理完成后保存一次（避免数据丢失）
            df.to_excel(output_path, index=False)
            logger.info(f"{sheet_name} 第 {batch_num} 批结果已临时保存至 {output_path}")
            
            # 批次间添加间隔（最后一批不添加）
            if batch_num < total_batches:
                wait_time = 10  # 批次间隔10秒（可根据API限流调整）
                logger.info(f"批次间隔 {wait_time} 秒，等待下一批处理...")
                time.sleep(wait_time)

        logger.info(f"表格 {sheet_name} 所有批次处理完成，最终结果保存至 {output_path}")

    logger.info(f"所有表格处理完成，结果保存在 {OUTPUT_DIR}")


# ========== 主入口 ==========
def main():
    if not API_KEY:
        logger.critical("请设置 API_KEY")
        return

    excel_file = Path(file_use)
    if not excel_file.exists():
        logger.error(f"Excel文件不存在: {excel_file}")
        return

    logger.info(f"开始处理Excel文件: {excel_file}")
    # 调用分批处理函数，每批20行（可根据需要调整）
    process_excel_file(excel_file, batch_size=20)
    logger.info("Excel处理完成")


if __name__ == "__main__":
    main()
