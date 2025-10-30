# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AgentOCR is a research repository combining two major projects for long-context processing:

1. **MemAgent**: A reinforcement learning framework that enables LLMs to process arbitrarily long contexts through a memory-based agent architecture
2. **Glyph**: A visual-text compression framework that renders long text as images for efficient processing by vision-language models (VLMs)

Both projects aim to overcome traditional context window limitations but use fundamentally different approaches.

## Environment Setup

### Initial Setup

```bash
# Create conda environment
conda create -n agentocr python=3.12
conda activate agentocr

# MemAgent dependencies
cd MemAgent
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm omegaconf
pip install -r requirements.txt
pip install --upgrade huggingface-hub==0.36.0

# Glyph dependencies
cd ../glyph
sudo apt-get install poppler-utils -y
pip install transformers==4.57.1 gradio pdf2image reportlab pdfplumber

# For downloading large models/datasets
sudo apt install git-lfs
git lfs install
```

### Environment Variables

```bash
export PYTHONPATH=$PYTHONPATH:/home/aiscuser/MemAgent:/home/aiscuser/AgentOCR/glyph/scripts
export DATAROOT=/home/aiscuser/AgentOCR/data/hotpotqa  # For MemAgent evaluation
export MODELROOT=/path/to/models  # For storing downloaded models
```

## MemAgent Architecture

### Core Concepts

MemAgent uses a multi-conversation RL framework where:
- Long contexts are split into chunks and processed sequentially
- A memory agent maintains state across chunks through a "memory" mechanism
- End-to-end optimization via Reinforcement Learning from Verifiable Rewards (RLVR)
- Supports context-independent multi-turn conversations (not just concatenated chat history)

### Key Components

- **Sync Mode** (`recurrent/impls/memory.py`): Multi-step workflow with independent contexts
- **Async Mode** (`recurrent/impls/async_*.py`): Agent-as-function paradigm allowing OpenAI-style API calls without boilerplate tensor operations
- **Generation Managers**: `recurrent/generation_manager.py` (sync) and `recurrent/async_generation_manager.py` (async)
- **Reward System**: Uses verifiers with different strictness levels for training vs. testing (see `verl/utils/reward_score/hotpotqa.py` for training, `taskutils/memory_eval/utils/__init__.py` for testing)

### Running MemAgent

**Quickstart (easiest way):**
```bash
cd MemAgent

# Local vLLM deployment
vllm serve BytedTsinghua-SIA/RL-MemoryAgent-14B --tensor_parallel_size 2
python quickstart.py --model BytedTsinghua-SIA/RL-MemoryAgent-14B

# Or with online LLM service
export URL=https://your-endpoint
export API_KEY=your-key
python quickstart.py --model gpt-4o-2024-11-20
```

**Evaluation:**
```bash
cd MemAgent/taskutils/memory_eval

# Deploy vLLM server first (in separate terminal)
vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor_parallel_size 8

# Run evaluation (uses all available GPUs)
python run.py

# For multi-node ray cluster
SERVE_PORT=8000 DASH_PORT=8265 python run.py

# Visualize results
python visualize.py
```

**Training:**
```bash
cd MemAgent

# Edit run_memory_7B.sh or run_memory_14B.sh to set:
# - PROJ_ROOT (for checkpoints)
# - DATASET_ROOT (training data location)

# Single-node training
bash run_memory_7B.sh

# Or configure ray cluster for multi-node training
```

### Data Preparation

```bash
cd MemAgent/taskutils/memory_data
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire

# Download datasets
bash download_qa_dataset.sh

# Process data for training
python processing.py  # Creates synthetic long context multi-hop QA

# Deploy models for filtering (in separate terminals)
# - Qwen-7B on localhost:8000
# - Qwen-7B-Instruct on localhost:8001

# Filter data
python filter.py -i hotpotqa_dev_process.parquet -o hotpotqa_dev_result --noresume
python filter.py -i hotpotqa_train_process.parquet -o hotpotqa_train_result --noresume
python filter2.py

# Create evaluation datasets
export DATAROOT="path/to/hotpotqa_dev.parquet"
python convert_to_eval.py  # Creates eval_200.json
python different_docs_eval.py  # Different document counts

# Create RULER evaluation datasets
python download_paulgraham_essay.py
bash ruler_data_prepare.sh
```

## Glyph Architecture

### Core Concepts

Glyph transforms long-context processing into a multimodal problem:
- Renders long text as images using configurable styles (fonts, colors, layout)
- Processes rendered images with vision-language models (VLMs)
- Achieves 3-4x compression with DPI=72, or 2-3x with DPI=96 (better quality)
- Built on GLM-4.1V-9B-Base, supports vLLM acceleration

### Key Components

- **Rendering**: `glyph/scripts/word2png_function.py` - Uses reportlab to convert text to PDF then images
- **Configuration**: `glyph/config/config_en.json` and `config_zh.json` - Control rendering parameters
- **Inference**: `glyph/scripts/vlm_inference.py` - VLM model inference on rendered images

### Running Glyph

**Demo:**
```bash
cd glyph/demo

# Compare Glyph vs baseline model
bash run_demo_compared.sh

# Glyph only
bash run_demo.sh
```

**Text to Image Rendering:**
```python
from word2png_function import text_to_images
import json

CONFIG_EN_PATH = 'glyph/config/config_en.json'
OUTPUT_DIR = 'output_images'
INPUT_FILE = 'data/hotpotqa/eval_50.json'

text = json.load(open(INPUT_FILE))[0]['context']

images = text_to_images(
    text=text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='test_001'
)
```

**Model Deployment:**
```bash
# Start vLLM server
vllm serve zai-org/Glyph --port 5002 --served-model-name glyph \
    --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}'
```

**Inference:**
```python
from vlm_inference import vlm_inference

response = vlm_inference(
    question="Based on the story in the figures, what is the ending?",
    image_paths=["./output_images/example/page_001.png"]
)
```

### Glyph Evaluation

```bash
cd glyph/evaluation

# Download evaluation data first
# Place in appropriate directories: longbench/data/, mrcr/data/, ruler/data/

# LongBench
python ./longbench/scripts/add_uid_jsonl.py --chose_newline
python ./longbench/scripts/word2png_longbench.py
python ./longbench/scripts/post_api_longbench.py --use_image
python ./longbench/scripts/clear_pred.py ./longbench/pred/glyph ./longbench/results
python ./longbench/scripts/eval_longbench.py

# MRCR
python ./mrcr/scripts/word2png_mrcr.py
python ./mrcr/scripts/post_api_mrcr.py

# RULER
python ./ruler/scripts/word2png_ruler.py
python ./ruler/scripts/post_api_ruler.py
```

## Architecture Notes

### MemAgent Engineering Highlights

1. **Multi-Conv RL Framework**: Extends DAPO algorithm to support end-to-end optimization of agent workflows with context-independent conversations (inspired by Search-R1)

2. **Agent-as-Function**: Novel framework allowing users to implement agents as functions calling LLMs in OpenAI-style without tensor operations boilerplate

3. **Ray Actor Process Pool**: CPU-intensive tasks (reward computation, tool calling) run asynchronously via Ray actors to avoid blocking the head node

4. **Chat Template Modifications**: Supports tool-response masking without manual tensor operations

### Glyph Rendering Configuration

Key parameters in `glyph/config/config_*.json`:

- `newline-markup`: Controls newline rendering
  - `"<font color=\"#FF0000\"> \\n </font>"`: Visual marker (higher compression)
  - `"<br/>"`: Standard line break
- `dpi`: Resolution setting (72 for best compression/speed, 96 for better quality)
- `font-size`, `page-width`, `page-height`: Layout parameters
- Colors: `page-bg-color`, `font-color`, `para-bg-color`

### Important Verification Notes

**MemAgent**: Training uses a stricter verifier (exact case matching, `\boxed{}` required) while testing uses relaxed verification (ignores articles, case, punctuation). This is intentional to prevent reward hacking during training. Expect ~50% validation score during training vs. ~80% final test score.

## Common Workflows

### Testing a Single MemAgent Task
```bash
cd MemAgent/taskutils/memory_eval
# Modify run.py to select specific test/model
# Set test_cases list to desired evaluation (e.g., specific RULER tasks)
python run.py
```

### Creating Custom Rendering Config for Glyph
1. Copy `glyph/config/config_en.json` to new file
2. Modify parameters (font, colors, DPI, spacing)
3. Test rendering: `python glyph/scripts/word2png_function.py` with custom config
4. Evaluate impact on compression ratio and accuracy

### Multi-Node Ray Cluster Setup
Both projects can use Ray for distributed execution:
```bash
# On head node
ray start --head --port=6379 --dashboard-port=8265

# On worker nodes
ray start --address='head-node-ip:6379'

# Run evaluation/training from head node
```

## Data Locations

- `/data/hotpotqa/`: Main QA evaluation datasets (eval_50.json, eval_100.json, etc.)
- `/data/Glyph_Evaluation/`: Pre-processed datasets for Glyph evaluation
  - `longbench/`: LongBench dataset
  - `mrcr/`: MRCR dataset
  - `ruler/`: RULER benchmark data
- `MemAgent/taskutils/memory_data/`: Training data and processing scripts

## Models

**MemAgent Models:**
- BytedTsinghua-SIA/RL-MemoryAgent-14B (recommended)
- BytedTsinghua-SIA/RL-MemoryAgent-7B
- Requires vLLM for serving

**Glyph Model:**
- zai-org/Glyph (based on GLM-4.1V-9B-Base)
- Download from HuggingFace

**Qwen Models** (for baselines/filtering):
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct
- Note: Must manually modify `config.json` to activate YaRN for long context
