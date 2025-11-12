# init environment called agentocr
```bash
# 1. setup environment
cd ~
rm -rf AgentOCR
git clone https://github.com/zihanwang314/OCR.git AgentOCR

conda create -n agentocr python=3.12 -y
conda activate agentocr

cd AgentOCR
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install "ray[serve,default]" httpx==0.23.1 aiohttp -U vllm omegaconf datasets --no-build-isolation
pip install -r MemAgent/requirements.txt
pip install --upgrade huggingface-hub==0.36.0

# 2. download memagent data
cd MemAgent/taskutils/memory_data
bash download_qa_dataset.sh
mkdir -p ../../../data
cd ../../../data
bash ../MemAgent/hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10
export DATAROOT=$(pwd)/hotpotqa
cd ..

# 3. install glyph dependencies
sudo apt-get install poppler-utils -y
pip install transformers==4.57.1 gradio pdf2image reportlab pdfplumber pyyaml

# 4. download glyph eval data
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/CCCCCC/Glyph_Evaluation data/glyph_eval

for f in data/glyph_eval/{mrcr,ruler}/data/*.json; do
  jq '[.[] | .config = {}]' "$f" > tmp && mv tmp "$f"
done
```





# Start to train and eval
```bash

# 5. deploy vllm to evaluate on MemAgent
conda activate agentocr

export DATAROOT=$(pwd)/hotpotqa
export PYTHONPATH=$PYTHONPATH:/home/aiscuser/AgentOCR/MemAgent:/home/aiscuser/AgentOCR/glyph/scripts
export DATAROOT=/home/aiscuser/AgentOCR/data/hotpotqa # for MemAgent
export RESULT_ROOT="."

cd MemAgent/taskutils/memory_eval
python run.py && cp -r results/* /blob/v-zihanwang/AgentOCR-results
python visualize.py
```

# evaluate ruler on glyph

```bash
conda activate agentocr
# render images
python glyph/evaluation/ruler/scripts/word2png_ruler.py --result-root results/dpi_48 --lens-list 4096,8192,16384,32768 --dpi 48
python glyph/evaluation/ruler/scripts/word2png_ruler.py --result-root results/dpi_64 --lens-list 4096,8192,16384,32768 --dpi 64
python glyph/evaluation/ruler/scripts/word2png_ruler.py --result-root results/dpi_72 --lens-list 4096,8192,16384,32768 --dpi 72
python glyph/evaluation/ruler/scripts/word2png_ruler.py --result-root results/dpi_96 --lens-list 4096,8192,16384,32768 --dpi 96
python glyph/evaluation/ruler/scripts/word2png_ruler.py --result-root results/dpi_120 --lens-list 4096,8192,16384,32768 --dpi 120


# start running evaluation
# MODEL_NAME=zai-org/Glyph
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL_NAME --tensor-parallel-size 8 --port 8001 --host 0.0.0.0
python glyph/evaluation/ruler/scripts/post_api_ruler.py --port 8001 --model-name $MODEL_NAME --result-root results/dpi_48 --lens-list 4096,8192,16384,32768
python glyph/evaluation/ruler/scripts/post_api_ruler.py --port 8001 --model-name $MODEL_NAME --result-root results/dpi_64 --lens-list 4096,8192,16384,32768
python glyph/evaluation/ruler/scripts/post_api_ruler.py --port 8001 --model-name $MODEL_NAME --result-root results/dpi_72 --lens-list 4096,8192,16384,32768
python glyph/evaluation/ruler/scripts/post_api_ruler.py --port 8001 --model-name $MODEL_NAME --result-root results/dpi_96 --lens-list 4096,8192,16384,32768
python glyph/evaluation/ruler/scripts/post_api_ruler.py --port 8001 --model-name $MODEL_NAME --result-root results/dpi_120 --lens-list 4096,8192,16384,32768

# find all things like results/dpi_xx/results/NUMBER/filename.json(l), save them to /blob/v-zihanwang/zai_org_glyph/dpi_xx/NUMBER/filename.json(l)
SRC=results
DEST=/blob/v-zihanwang/zai_org_glyph

for f in $(find "$SRC" -type f \( -name "evaluation.json" -o -name "evaluation_summary.txt" -o -name "predictions.jsonl" \)); do
  rel=${f#$SRC/}                         # 去掉前缀 results/
  dest="$DEST/${rel#*/}"                 # 去掉 dpi_xx 前的部分，保持层级 dpi_xx/NUMBER/...
  mkdir -p "$(dirname "$dest")"
  if [[ $f == *predictions.jsonl ]]; then
    head -n 10 "$f" > "$dest"            # predictions.jsonl 截前 10 行
  else
    cp "$f" "$dest"                      # 其他文件直接复制
  fi
  echo "→ $dest"
done

```



```python
from word2png_function import text_to_images
import json

CONFIG_EN_PATH = 'Glyph/config/config_en.json'
OUTPUT_DIR = 'output_images'
INPUT_FILE = 'data/hotpotqa/eval_50.json'

text = json.load(open(INPUT_FILE))[0]['context']

# Convert text to images
images = text_to_images(
    text=text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='test_001'
)

print(f"\nGenerated {len(images)} image(s):")
for img_path in images:
    print(f"  {img_path}")
```

inference:
```python
import requests
import json
A = json.load(open("post_api_ruler_error.json"))
api_url, data, headers, timeout = A['api_url'], A['data'], A['headers'], A['timeout']
data = json.dumps(data)
response = requests.post(api_url, data=data, headers=headers, timeout=timeout)
response
```

# get results
python run.py