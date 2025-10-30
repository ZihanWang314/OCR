import os
import json
import base64
import io
import math
from typing import List, Optional, Union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import requests
import argparse
import os

# —— 原有的 API 调用和图片编码函数 —— #

model_name = "glyph"
# api_key = "EMPTY" 

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--use_image', action='store_true', help="Use image processing")
    parser.add_argument('--api_url', type=str, default="http://your_api_url:port/v1/chat/completions", help="API endpoint URL")
    parser.add_argument('--pool_size', type=int, default=8, help="Process pool size")
    parser.add_argument('--input_dir', type=str, default='./longbench/rendered_images', help="Input directory for dataset files")
    parser.add_argument('--output_dir', type=str, default='./longbench/pred', help="Output directory for prediction files")
    parser.add_argument('--model_name', type=str, default='glyph', help="Model name for creating output subdirectory")
    return parser.parse_args(args)

def encode_image_with_max_pixels(image_path: str, max_pixels: int = 100000, save_compressed=False) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        if save_compressed:
            base, ext = os.path.splitext(image_path)
            im.save(f"{base}_compressed.png", format="PNG")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
 
        
def post_api(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    max_pixels: int = 36000000,
    model: str = "doubao-1-5-thinking-vision-pro-250428",
    temperature: float = 0.7,
    api_url: str = "http://your.api.endpoint/v1/chat/completions",
) -> Union[str, None]:
    """
    自定义请求逻辑，兼容你已有的 JSON 构造方式。
    """
    messages = []
    # 清理prompt中可能的换行符和多余空格
    prompt = prompt.strip() if prompt else ""
    conversation_history = [{"role": "user", "content": prompt}]
    model_config = {
        "max_tokens": 8192,
        "top_p": 1.0,
        "top_k": 1,
        "temperature": 0.0001,
        "repetition_penalty": 1.1
    }

    for i, msg in enumerate(conversation_history):
        if msg["role"] == "user":
            user_contents = []

            if i == 0 and image_paths:
                for image_path in image_paths:
                    image_path = image_path.strip()  # 清理路径
                    if os.path.exists(image_path):
                        encoded = encode_image_with_max_pixels(image_path, max_pixels=max_pixels)
                        user_contents.append({
                            'type': 'image_url',
                            'image_url': {"url": f"data:image/png;base64,{encoded}"}
                        })

            user_contents.append({'type': 'text', 'text': msg["content"]})
            messages.append({'role': 'user', 'content': user_contents})

        elif msg["role"] == "assistant":
            messages.append({'role': 'assistant', 'content': msg["content"]})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": model_config["max_tokens"],
        "top_p": model_config["top_p"],
        "top_k": model_config["top_k"],
        "temperature": model_config["temperature"],
        "repetition_penalty": model_config["repetition_penalty"],
        "skip_special_tokens": False,
        "stop_token_ids": [151329, 151348, 151336],
        "include_stop_str_in_output": True
    }

    headers = {
        'Content-Type': 'application/json',
        # 'Authorization': f'Bearer {api_key}'
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=1200)
        
        # 首先检查HTTP状态码
        if response.status_code != 200:
            error_msg = f"HTTP error: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f", Response: {error_detail}"
            except:
                error_msg += f", Raw response: {response.text}"
            print(error_msg)
            return None
        
        # 解析JSON响应
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}, Raw response: {response.text}")
            return None
        
        # 检查响应结构
        if 'choices' not in result:
            print(f"API response missing 'choices' field. Full response: {result}")
            return None
            
        if not result['choices'] or len(result['choices']) == 0:
            print(f"API response has empty 'choices' field. Full response: {result}")
            return None
            
        if 'message' not in result['choices'][0]:
            print(f"API response missing 'message' field in choices[0]. Full response: {result}")
            return None
            
        if 'content' not in result['choices'][0]['message']:
            print(f"API response missing 'content' field in message. Full response: {result}")
            return None
        
        # 检查usage字段（可选）
        usage = result.get('usage', {})
        
        return result['choices'][0]['message']['content'], usage['prompt_tokens']
        
    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return None


# —— 新增：单条数据处理函数 —— #

def debug_path_issue(path_str):
    """调试路径问题的辅助函数"""
    if isinstance(path_str, str):
        # 检查是否有隐藏字符
        repr_path = repr(path_str)
        if '\\n' in repr_path or '\\r' in repr_path or '\\t' in repr_path:
            print(f"发现路径中有隐藏字符: {repr_path}")
            return path_str.strip().replace('\n', '').replace('\r', '').replace('\t', '')
    return path_str

def process_item(args_tuple) -> dict:
    item, use_image, api_url, dataset2prompt = args_tuple
    try:
        if use_image:
            prompt = dataset2prompt[item['dataset']].replace('{input}', item['input']).strip()
            image_paths_data = item['image_paths']
            
            # 调试路径问题
            image_paths_data = debug_path_issue(image_paths_data)
            
            full_paths = []
            # 检查 image_paths_data 是目录（str）还是文件列表（list）
            if isinstance(image_paths_data, str) and os.path.isdir(image_paths_data.strip()):
                # 如果是目录，则列出所有文件
                clean_dir = image_paths_data.strip()
                filenames = sorted(os.listdir(clean_dir))
                full_paths = [os.path.join(clean_dir, fn) for fn in filenames]
            elif isinstance(image_paths_data, list):
                # 如果已经是列表，直接使用（清理每个路径）
                full_paths = [debug_path_issue(path) if isinstance(path, str) else path for path in image_paths_data]
            else:
                print(f"Warning: Invalid image_paths format for {item['_id']}: {repr(image_paths_data)}")

            # 过滤掉不存在的文件路径
            existing_paths = [p for p in full_paths if os.path.exists(p)]
            if len(existing_paths) != len(full_paths):
                print(f"Warning: Some image paths not found for {item['_id']}")

            result = post_api(prompt, api_url=api_url, image_paths=existing_paths, max_pixels=36000000)
        else:
            prompt = dataset2prompt[item['dataset']].replace('{context}', item['context']).replace('{input}', item['input']).strip()
            result = post_api(prompt, api_url=api_url)

        if result is None:
            return None
            
        response, usage = result

        # 按照 pred.py 的格式构建输出
        return {
            "pred": response,
            "answers": item["answers"],
            "all_classes": item["all_classes"],
            "length": item["length"],
            "usage": str(usage),
        }
    except Exception as e:
        error_msg = f"Error processing {item.get('_id', 'unknown')}: {type(e).__name__}: {e}"
        # 如果是路径相关错误，提供更详细的信息
        if 'path' in str(e).lower() or '路径' in str(e) or 'workspace/zx' in str(e):
            error_msg += f"\n问题数据: {item}"
        print(error_msg)
        return None

# —— 主程序，多进程并行 —— #

if __name__ == "__main__":
    args = parse_args()
    
    # 确定要处理的数据集
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                   "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                   "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                   "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # 加载prompt配置
    if args.use_image:
        print("使用图像处理，加载 dataset2vlmprompt.json 配置文件")
        with open('./longbench/config/dataset2vlmprompt.json', encoding='utf-8') as f:
            dataset2prompt = json.load(f)
    else:
        with open('./longbench/config/dataset2prompt.json', encoding='utf-8') as f:
            dataset2prompt = json.load(f)
    

    model_name = args.model_name
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(f"{args.output_dir}/{model_name}"):
        os.makedirs(f"{args.output_dir}/{model_name}")

    # 处理每个数据集
    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset}")
        
        input_file = f'{args.input_dir}/{dataset}.jsonl'
        output_file = f'{args.output_dir}/{model_name}/{dataset}.jsonl'
        
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在: {input_file}")
            continue
        
        # 如果输出文件存在，先删除
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"{output_file} 已删除。")

        # 读取已完成的数据ID（如果有的话）
        exist_ids = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            exist_ids.add(json.loads(line)['_id'])
            except Exception as e:
                print(f"读取已完成数据时出错: {e}")
        
        # 读取待处理数据
        all_data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            # 处理数据集名称
                            if item['dataset'].endswith('_e'):
                                item['dataset'] = item['dataset'][:-2]
                            
                            # 清理可能存在的换行符和空格
                            if 'input' in item and isinstance(item['input'], str):
                                item['input'] = item['input'].strip()
                            if 'context' in item and isinstance(item['context'], str):
                                item['context'] = item['context'].strip()
                            if 'image_paths' in item:
                                if isinstance(item['image_paths'], str):
                                    item['image_paths'] = debug_path_issue(item['image_paths'])
                                elif isinstance(item['image_paths'], list):
                                    item['image_paths'] = [debug_path_issue(path) if isinstance(path, str) else path 
                                                         for path in item['image_paths']]
                                
                            all_data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"跳过第{line_num}行无效JSON: {e}")
        except Exception as e:
            print(f"读取输入文件时出错: {e}")
            continue
        
        # 过滤掉已完成的数据
        to_process = [d for d in all_data if d.get('_id') not in exist_ids]
        
        if not to_process:
            print(f"数据集 {dataset} 没有需要处理的数据")
            continue
        
        print(f"数据集 {dataset}: 总计 {len(all_data)} 条数据，需要处理 {len(to_process)} 条")
        
        # 准备多进程参数
        process_args = [(item, args.use_image, args.api_url, dataset2prompt) for item in to_process]
        
        # 启动进程池处理
        total_tasks = len(to_process)
        success_count = 0
        error_count = 0
        
        with Pool(processes=args.pool_size) as pool:
            with open(output_file, 'a', encoding='utf-8') as fout:
                # 使用 imap_unordered 进行并行处理
                for result in tqdm(pool.imap_unordered(process_item, process_args), 
                                 total=total_tasks, 
                                 desc=f"Processing {dataset}"):
                    if result is not None:
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1
                    else:
                        error_count += 1
        
        print(f"数据集 {dataset} 处理完成: 成功 {success_count} 条，失败 {error_count} 条")
    
    print("\n所有数据集处理完成。")
