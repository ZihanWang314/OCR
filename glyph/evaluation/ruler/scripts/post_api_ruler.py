import json
import requests
import os
import argparse
import re
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
from PIL import Image
import math
import io
import base64

env_var_name1 = "http_proxy"
env_var_name2 = "https_proxy"
if env_var_name1 in os.environ:
    os.environ.pop(env_var_name1)
    print(f"Environment variable '{env_var_name1}' has been unset.")
else:
    print(f"Environment variable '{env_var_name1}' does not exist.")
if env_var_name2 in os.environ:
    os.environ.pop(env_var_name2)
    print(f"Environment variable '{env_var_name2}' has been unset.")
else:
    print(f"Environment variable '{env_var_name2}' does not exist.")

headers = {'Content-Type': 'application/json'}
URL = 'http://your_api_url:port/v1/chat/completions'
MAX_PIXELS = 36000000

def encode_image_with_max_pixels(image_path: str, max_pixels: int) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB"); w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode("utf-8")

# RULER 评分函数
def string_match_part(preds, refs):
    """部分匹配评分：只要预测结果中包含任意一个正确答案就得1分"""
    recalls = [max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]
    score = sum(recalls) / len(preds) * 100
    return round(score, 2), recalls

def string_match_all(preds, refs):
    """全部匹配评分：计算预测结果中包含的正确答案比例"""
    recalls = [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]
    score = sum(recalls) / len(preds) * 100
    return round(score, 2), recalls

# RULER 任务配置
tasks_base = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are""",
        'metric_fn': string_match_all,
    },
    
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """,
        'metric_fn': string_match_all,
    },
    
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:""",
        'metric_fn': string_match_all,
    },
    
    'freq_words_extraction' : {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:""",
        'metric_fn': string_match_all,
    },

    'qa': {
        'tokens_to_generate': 32, 
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
        'metric_fn': string_match_part,
    },
}

tasks_customized = {
    'niah_single_1': {'task': 'niah'},
    'niah_single_2': {'task': 'niah'},
    'niah_single_3': {'task': 'niah'},
    'niah_multikey_1': {'task': 'niah'},
    'niah_multikey_2': {'task': 'niah'},
    'niah_multikey_3': {'task': 'niah'},
    'niah_multivalue': {'task': 'niah'},
    'niah_multiquery': {'task': 'niah'},
    'vt': {'task': 'variable_tracking'},
    'cwe': {'task': 'common_words_extraction'},
    'fwe': {'task': 'freq_words_extraction'},
    'qa_1': {'task': 'qa'},
    'qa_2': {'task': 'qa'}
}

def postprocess_pred(predict_str: str):
    """后处理预测结果"""
    predict_str = predict_str.strip()
    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str

def evaluate_single_task(task_name, predictions, references):
    """评估单个任务"""
    if task_name not in tasks_customized:
        print(f"警告：未知任务 {task_name}")
        return 0.0, []
    
    # 获取任务配置
    task_config = tasks_customized[task_name].copy()
    base_task = task_config['task']
    if base_task in tasks_base:
        task_config.update(tasks_base[base_task])
    
    # 后处理预测结果
    processed_preds = [postprocess_pred(pred) for pred in predictions]
    
    # 计算评分
    if 'metric_fn' in task_config and callable(task_config['metric_fn']):
        score, recalls = task_config['metric_fn'](processed_preds, references)
        return score, recalls
    else:
        print(f"警告：任务 {task_name} 没有有效的评分函数")
        return 0.0, []

def process_data_item(data_item, model_name="glm-4v"):
    """处理单个数据项，构建发送给API的消息"""
    # 构建text内容：example字段拼接question字段（处理example可能不存在的情况）
    text_parts = []
    # if 'example' in data_item and data_item['example']:
    #     text_parts.append(str(data_item['example']))
    if 'question' in data_item and data_item['question']:
        text_parts.append(str(data_item['question']))
    
    text = " ".join(text_parts)
    
    # 构建content列表
    content = []
    

    
    # 处理图片列表，添加图片内容
    if 'image_paths' in data_item and isinstance(data_item['image_paths'], list):
        for img_path in data_item['image_paths']:
            base64_image = encode_image_with_max_pixels(img_path, MAX_PIXELS)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            else:
                print(f"警告：图片路径不存在或无法编码 - {img_path}")


        # 添加文本内容
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    # 构建完整的OpenAI格式消息结构
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    message = {
        "messages": messages
    }
    
    return message

def send_to_api(api_url, message, headers=None, max_retries=3):
    """发送消息到API，带重试机制"""
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    
    data = {
        "model": "glyph",
        "messages": message['messages'],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": False,
        "max_tokens": 8192,
        # "top_p": 1.0,
        "top_k": 50, # 50
        "temperature": 0.0, # 0.7
        "repetition_penalty": 1.1,
        "stop_token_ids": [151329, 151348, 151336]
        # force_thinking 
    }
    
    for attempt in range(max_retries):
        try:
            # 根据重试次数增加超时时间
            timeout = 180 + (attempt * 60)  # 180秒开始，每次重试增加60秒
            response = requests.post(api_url, data=json.dumps(data), headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout as e:
            print(f"第 {attempt + 1} 次请求超时 (timeout={timeout}s): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 等待5秒、10秒、15秒
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"所有重试均失败，放弃请求")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"第 {attempt + 1} 次请求出错: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"所有重试均失败，放弃请求")
                return None
    
    return None

def process_single_item(api_url, data_item, index, model_name, headers, temp_dir, lock):
    """处理单个数据项的多线程函数"""
    try:
        print(f"处理第 {index+1} 个数据项")
        message = process_data_item(data_item, model_name)
        if not message:
            return None
            
        response = send_to_api(api_url, message, headers)
        
        # 提取预测结果
        pred_text = ""
        if response and 'choices' in response and len(response['choices']) > 0:
            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                pred_text = response['choices'][0]['message']['content']
                
                # 过滤结尾的 <|user|> 标签
                if pred_text.endswith('<|user|>'):
                    pred_text = pred_text[:-8].rstrip()
                
                # 检查是否包含 <think> 标签，如果有则提取 </think> 之后的内容
                # if '</think>' in raw_content:
                #     pred_text = raw_content.split('</think>')[-1].strip()
                # else:
                #     pred_text = ""
        
        # 获取任务名称和参考答案
        task_name = data_item.get('task_name', 'unknown')
        references = data_item.get('outputs', [])
        if isinstance(references, str):
            references = [references]
        
        result = {
            'index': index,
            'task_name': task_name,
            'prediction': pred_text,
            'references': references,
            'message': message,
            'response': response,
            'timestamp': time.time()
        }
        
        # 写入临时文件（线程安全）
        # temp_file = os.path.join(temp_dir, f"result_{index:06d}.json")
        # with lock:
        #     with open(temp_file, 'w', encoding='utf-8') as f:
        #         json.dump(result, f, ensure_ascii=False, indent=2)
        
        # print(f"第 {index+1} 个数据项处理完成，结果已写入 {temp_file}")
        return result
        
    except Exception as e:
        print(f"处理第 {index+1} 个数据项时出错: {e}")
        return None

def main(jsonl_file_path, api_url, model_name="glm-4v", headers=None, max_workers=5):
    """主函数：读取JSON文件，使用多线程处理数据并发送到API，并进行评分"""
    try:
        # 读取JSON文件
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f if line.strip()]
        
        if not isinstance(data_list, list):
            print("错误：JSON文件内容不是一个列表")
            return
        
        print(f"共有 {len(data_list)} 个数据项需要处理，使用 {max_workers} 个线程")
        
        # 创建临时目录存储中间结果
        temp_dir = "" # if you want to specify a temp dir, set it here
        os.makedirs(temp_dir, exist_ok=True)
        print(f"临时结果将保存到: {temp_dir}")
        
        # 线程锁，用于文件写入的线程安全
        lock = threading.Lock()
        
        # 使用线程池处理所有数据项
        results = []
        task_results = defaultdict(lambda: {'predictions': [], 'references': [], 'indices': []})
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    process_single_item, 
                    api_url, 
                    data_item, 
                    i, 
                    model_name, 
                    headers, 
                    temp_dir, 
                    lock
                ): i for i, data_item in enumerate(data_list)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # 存储结果用于评分
                        task_name = result['task_name']
                        task_results[task_name]['predictions'].append(result['prediction'])
                        task_results[task_name]['references'].append(result['references'])
                        task_results[task_name]['indices'].append(result['index'])
                        
                    print(f"进度: {completed}/{len(data_list)} 完成")
                    
                except Exception as e:
                    print(f"处理第 {index+1} 个数据项时出错: {e}")
        
        # 按index排序结果
        results.sort(key=lambda x: x['index'])
        
        # 对每个任务进行评分
        evaluation_results = {}
        summary_lines = []
        total_score = 0
        total_valid = 0
        
        print("\n开始评分...")
        for task_name, task_data in task_results.items():
            # 跳过指定的任务
            if task_name in ['niah_single_3', 'niah_multikey_3']:
                print(f"跳过任务: {task_name}")
                continue

            if len(task_data['predictions']) > 0:
                score, recalls = evaluate_single_task(
                    task_name, 
                    task_data['predictions'], 
                    task_data['references']
                )
                evaluation_results[task_name] = {
                    'score': score,
                    'valid': len(task_data['predictions']),
                    'recalls': recalls
                }
                total_score += score
                total_valid += len(task_data['predictions'])

                task_summary = f"任务 {task_name}: 得分 {score:.2f}, 有效样本 {len(task_data['predictions'])}"
                print(task_summary)
                summary_lines.append(task_summary)
        
        # 计算总体平均分
        if len(evaluation_results) > 0:
            avg_score = total_score / len(evaluation_results)
            evaluation_results['average'] = {
                'score': avg_score,
                'valid': total_valid
            }
            overall_summary = f"总体平均分: {avg_score:.2f} (有效样本总数 {total_valid})"
            print(f"\n{overall_summary}")
            summary_lines.append("")
            summary_lines.append(overall_summary)
        
        print("所有数据处理和评分完成")
        print(f"临时结果文件保存在: {temp_dir}")
        
        return {
            'results': results,
            'evaluation': evaluation_results,
            'summary_lines': summary_lines,
            'temp_dir': temp_dir
        }
        
    except FileNotFoundError:
        print(f"错误：找不到JSON文件 - {jsonl_file_path}")
    except json.JSONDecodeError:
        print(f"错误：JSON文件解析失败 - {jsonl_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    
    return None

if __name__ == "__main__":
    # 定义要遍历的长度列表
    all_lens = [4096, 8192, 16384, 32768, 65536, 126000]

    for lens in all_lens:
        print(f"--- 开始处理长度: {lens} ---")

        # 定义输出目录并检查是否存在
        output_dir = f"./ruler/results/{lens}"
        if os.path.exists(output_dir):
            print(f"输出目录 {output_dir} 已存在，跳过处理。")
            continue

        # 定义输入文件路径
        jsonl_file = f"./ruler/data/final_dpi96_processed_ruler_all_tasks_{lens}.jsonl"
        
        # 检查输入文件是否存在
        if not os.path.exists(jsonl_file):
            print(f"输入文件 {jsonl_file} 不存在，跳过处理长度 {lens}。")
            continue

        # 调用主函数
        result = main(jsonl_file, URL, model_name="glyph", headers=headers, max_workers=16)
        
        if result:
            # 创建输出目录
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 保存预测结果
            with open(os.path.join(output_dir, "predictions.jsonl"), 'w', encoding='utf-8') as f:
                for item in result['results']:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            # 保存评分结果
            with open(os.path.join(output_dir, "evaluation.json"), 'w', encoding='utf-8') as f:
                json.dump(result['evaluation'], f, ensure_ascii=False, indent=2)

            if result['summary_lines']:
                summary_path = os.path.join(output_dir, "evaluation_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(result['summary_lines']))
                print(f"评分汇总: {summary_path}")
            
            print(f"\n长度 {lens} 的最终结果已保存到 {output_dir}")
            print(f"预测结果: {os.path.join(output_dir, 'predictions.jsonl')}")
            print(f"评分结果: {os.path.join(output_dir, 'evaluation.json')}")
            print(f"临时结果文件在: {result['temp_dir']}")
        
        print(f"--- 长度 {lens} 处理完成 ---\n")

def test_evaluation():
    """测试评分功能"""
    # 测试数据
    test_preds = ["The answer is 42", "I don't know", "The value is 42 and 7"]
    test_refs = [["42"], ["unknown"], ["42", "7"]]
    
    print("测试部分匹配评分:")
    score, recalls = string_match_part(test_preds, test_refs)
    print(f"得分: {score}, 召回率: {recalls}")
    
    print("\n测试全部匹配评分:")
    score, recalls = string_match_all(test_preds, test_refs)
    print(f"得分: {score}, 召回率: {recalls}")
    
    print("\n测试单任务评分:")
    score, recalls = evaluate_single_task("qa_1", test_preds, test_refs)
    print(f"QA任务得分: {score}, 召回率: {recalls}")

# 如果需要测试评分功能，取消注释下面这行
# test_evaluation()
