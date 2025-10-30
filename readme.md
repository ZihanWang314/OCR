# init environment called agentocr
conda create -n agentocr python=3.12
conda activate agentocr

git clone 

cd MemAgent
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm omegaconf
pip install -r requirements.txt
pip install --upgrade huggingface-hub==0.36.0

cd ../Glyph
sudo apt-get install poppler-utils -y
pip install transformers==4.57.1 gradio pdf2image reportlab pdfplumber pyyaml

sudo apt install git-lfs
git lfs install
cd ../data
git clone https://huggingface.co/datasets/CCCCCC/Glyph_Evaluation

export PYTHONPATH=$PYTHONPATH:/home/aiscuser/MemAgent:/home/aiscuser/AgentOCR/Glyph/scripts
export DATAROOT=/home/aiscuser/AgentOCR/data/hotpotqa # for MemAgent


# deploy vllm to evaluate on MemAgent
conda activate agentocr

cd MemAgent
vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor_parallel_size 8



cd taskutils/memory_eval
python run.py
python visualize.py

# evaluate ruler on Glyph

python glyph/evaluation/ruler/scripts/word2png_ruler.py # render images

python glyph/evaluation/ruler/scripts/post_api_ruler.py # start running evaluation


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



# get results
python run.py