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
pip install transformers==4.57.1 gradio pdf2image reportlab

sudo apt install git-lfs
git lfs install
cd ../data
git clone https://huggingface.co/datasets/CCCCCC/Glyph_Evaluation


# deploy vllm to evaluate on MemAgent
conda activate agentocr

cd MemAgent
vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor_parallel_size 8

export PYTHONPATH=$PYTHONPATH:/home/aiscuser/MemAgent
export DATAROOT=/home/aiscuser/MemAgent/hotpotqa

cd taskutils/memory_eval
python run.py
python visualize.py

# evaluate on Glyph

export PYTHONPATH=$PYTHONPATH:/home/aiscuser/AgentOCR/Glyph/scripts

```python
from word2png_function import text_to_images

```



# get results
python run.py