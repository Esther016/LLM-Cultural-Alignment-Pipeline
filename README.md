# 🚀 Cross-Cultural LLM Orchestration Pipeline  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Status](https://img.shields.io/badge/Status-Research_Prototype-green)  
![Focus](https://img.shields.io/badge/Focus-High_Throughput_Inference-orange)  

## 📖 Abstract  

This repository hosts a robust **Batch Inference Orchestrator** designed for Computational Social Science research. It facilitates high-concurrency interactions with heterogeneous Large Language Models (LLMs) via the Aihubmix aggregator.  

The framework was engineered to support **comparative analysis studies**, specifically enabling simultaneous bilingual (English & Chinese) querying across diverse model architectures (e.g., GPT-4, Claude, Qwen, Llama) to quantify cross-cultural alignment and geopolitical and ideological biases.  

## ✨ Key Features  

* **⚡ High-Concurrency Architecture**: Utilizes `ThreadPoolExecutor` for asynchronous I/O, allowing parallel processing of large-scale datasets significantly faster than sequential methods.  
* **🌐 Bilingual Alignment Support**: Native handling of paired prompts (`Prompt_EN` / `Prompt_CN`), ensuring strict correspondence for cross-lingual evaluation tasks.  
* **🛡️ Robustness**:  
    * **Automatic Retry Logic**: Implements exponential backoff strategies to handle API rate limits and transient network failures.  
    * **Data Sanitization**: Specialized pre-processing to handle Excel-incompatible characters (via `ILLEGAL_CHARACTERS_RE`) ensuring data integrity.  
    * **State Persistence**: Real-time intermediate saving prevents data loss during long-running batch jobs.  
* **🔌 Model Agnostic**: Seamless switching between proprietary models (OpenAI, Gemini) and open-weights models (Llama 3, Qwen) via unified API routing.  

## 🛠️ System Architecture  

```text
├── llm_aihubmix.py                     # Core Orchestration Engine
├── requirements.txt                    # Dependency Manifest
├── Data Sample/                        # Sample Dataset Folder
│   ├── AllQuestions-sample-politics.xlsx
│   └── AllQuestions-sample-personality.xlsx
├── .env.example                        # Configuration Template
├── .gitignore                          # Security Rules
└── outputs/                            # Structured Data Lake (Auto-generated)
    └── AllQuestions_aihub_temp0.7.xlsx
```

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- A valid API Key from Aihubmix (or compatible OpenAI-format provider)  

### Installation  

#### Clone the Repository  
```bash
git clone https://github.com/Esther016/LLM-Cultural-Alignment-Pipeline.git
cd LLM-Cultural-Alignment-Pipeline
```

#### Environment Setup  
Install dependencies via the manifest file:  
```bash
pip install -r requirements.txt
```

#### Security Configuration  
Create a `.env` file to securely store credentials (**never commit this file**):  
```env
# .env file (add to .gitignore!)
AIHUBMIX_API_KEY=sk-xxxxxxxxxxxxxxxxx
```

## 📊 Usage Pipeline  

### 1. Data Preparation  
Prepare an input Excel file (default: `AllQuestions.xlsx`) with the following schema to support comparative studies:  

| Prompt               | Prompt_CN           | Question             | Question_CN          |  
|----------------------|---------------------|----------------------|----------------------|  
| System instruction...| 系统指令...         | Input question...    | 输入问题...          |  

### 2. Execution  
Run the orchestrator with a specified temperature parameter to control generation stochasticity:  
```bash
# Syntax: python script_name.py [temperature]
python llm_aihubmix.py 0.7
```  
- **Temperature**: Float (0.0–1.0). Lower values (e.g., 0.2) for factual extraction; higher values (e.g., 0.7) for creative/ideological simulations.  

### 3. Output Analysis  
Results are automatically aggregated into the `outputs/` directory. The engine generates comparative columns for each model:  
- `{Model_Name}`: English Response  
- `{Model_Name}_CN`: Chinese Response  

## ⚙️ Advanced Configuration  

Researchers can fine-tune pipeline parameters directly in the script for experimental customization:  

| Parameter       | Default | Description                                                                 |  
|-----------------|---------|-----------------------------------------------------------------------------|  
| `MODELS`        | [List]  | Array of model identifiers to test (e.g., `gpt-4o`, `claude-3-5-sonnet`).   |  
| `BATCH_SIZE`    | 20      | Number of rows processed before forcing a disk save.                        |  
| `MAX_WORKERS`   | 5       | Thread pool size. Increase for higher throughput (monitor API rate limits).  |  
| `TIMEOUT`       | 120s    | Max wait time per API call before triggering retry logic.                   |  

## 📂 Dataset & Privacy Note

**Note on Data Availability:**
This repository contains the **automation framework** developed for the research project. Due to ongoing publication processes and confidentiality agreements, the full dataset (2000+ questions) and the generated inference results are **not included**.

A `sample_dataset.xlsx` structure is provided in the `data/` folder for demonstration purposes. It contains the required schema:
* **Sheet 1**: `Politics` (Columns: `English Question`, `Chinese Question`)
* **Sheet 2**: `Personality` (Columns: `English Question`, `Chinese Question`)
## 🤝 Contribution  

This tool was originally developed for undergraduate research in Data Science & Political Science Alignment. Contributions to improve scheduler efficiency or add visualization modules are welcome:  

1. Fork the Project  
2. Create your Feature Branch (`git checkout -b feature/Optimization`)  
3. Commit your Changes (`git commit -m 'Add visualization module'`)  
4. Push to the Branch  
5. Open a Pull Request  

## 📜 Disclaimer  

This tool is for academic research purposes only. Users are responsible for adhering to the Terms of Service of respective LLM providers.  

---  


*Designed for reproducible computational social science — enabling rigorous cross-cultural LLM analysis at scale.*