import json
import requests
import os
import argparse
import re
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from PIL import Image
import io
import random
import time
from difflib import SequenceMatcher


headers = {'Content-Type': 'application/json'}
URL = 'http://your_api_url:port/v1/chat/completions'
MAX_PIXELS = 36000000

def grade_response(response, answer, random_string_to_prepend):
    """评分函数，使用SequenceMatcher计算相似度"""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    # from IPython import embed; embed()
    # 移除前缀后进行比较
    response_clean = response[len(random_string_to_prepend):].replace('<|user|>', '')
    answer_clean = answer[len(random_string_to_prepend):] if answer.startswith(random_string_to_prepend) else answer
    
    return float(SequenceMatcher(None, response_clean, answer_clean).ratio())

def encode_image_with_max_pixels(image_path: str, max_pixels: int) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def postprocess_pred(predict_str: str):
    """后处理预测结果"""
    predict_str = predict_str.strip()
    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str

def extract_category_from_unique_id(unique_id):
    """从unique_id中提取类别"""
    if "2needle_8k_16k" in unique_id:
        return "2needle_8k_16k"
    elif "2needle_0k_8k" in unique_id:
        return "2needle_0k_8k"
    elif "2needle_16k_32k" in unique_id:
        return "2needle_16k_32k"
    elif "2needle_32k_64k" in unique_id:
        return "2needle_32k_64k"
    elif "2needle_64k_128k" in unique_id:
        return "2needle_64k_128k"
    elif "2needle_128k_256k" in unique_id:
        return "2needle_128k_256k"
    elif "2needle_256k_512k" in unique_id:
        return "2needle_256k_512k"
    elif "2needle_512k_1024k" in unique_id:
        return "2needle_512k_1024k"
    elif "4needle_8k_16k" in unique_id:
        return "4needle_8k_16k"
    elif "4needle_0k_8k" in unique_id:
        return "4needle_0k_8k"
    elif "4needle_16k_32k" in unique_id:
        return "4needle_16k_32k"
    elif "4needle_32k_64k" in unique_id:
        return "4needle_32k_64k"
    elif "4needle_64k_128k" in unique_id:
        return "4needle_64k_128k"
    elif "4needle_128k_256k" in unique_id:
        return "4needle_128k_256k"
    elif "4needle_256k_512k" in unique_id:
        return "4needle_256k_512k"
    elif "4needle_512k_1024k" in unique_id:
        return "4needle_512k_1024k"
    elif "8needle_8k_16k" in unique_id:
        return "8needle_8k_16k"
    elif "8needle_0k_8k" in unique_id:
        return "8needle_0k_8k"
    elif "8needle_16k_32k" in unique_id:
        return "8needle_16k_32k"
    elif "8needle_32k_64k" in unique_id:
        return "8needle_32k_64k"
    elif "8needle_64k_128k" in unique_id:
        return "8needle_64k_128k"
    elif "8needle_128k_256k" in unique_id:
        return "8needle_128k_256k"
    elif "8needle_256k_512k" in unique_id:
        return "8needle_256k_512k"
    elif "8needle_512k_1024k" in unique_id:
        return "8needle_512k_1024k"
    else:
        return "unknown"

def evaluate_single_item(prediction, answer, random_string_to_prepend):
    """评估单个数据项"""
    score = grade_response(prediction, answer, random_string_to_prepend)
    return score

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
            if os.path.exists(img_path):
                encoded = encode_image_with_max_pixels(img_path, max_pixels=MAX_PIXELS)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}"
                    }
                })
            else:
                print(f"警告：图片路径不存在 - {img_path}")

        # 添加文本内容
    if text:
        content.append({
            "type": "text",
            "text": "\n"+text
        })
    
    # 构建完整的OpenAI格式消息结构
    message = {
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    return message

def send_to_api(api_url, message, headers=None, max_retries=3):
    """发送消息到API，带重试机制"""
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    # from IPython import embed; embed()
    data = {
        "model": "glyph",
        "messages": message['messages'],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": False,
        "max_tokens": 8192,
        "top_p": 1.0,
        "top_k": 50,
        "temperature": 0.0001,
        "repetition_penalty": 1.1,
        "stop_token_ids": [151329, 151348, 151336]
    }
    
    for attempt in range(max_retries):
        try:
            # 根据重试次数增加超时时间
            timeout = 600 + (attempt * 120)  # 180秒开始，每次重试增加60秒
            response = requests.post(api_url, data=json.dumps(data), headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # from IPython import embed; embed(); exit()
            
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
                raw_content = response['choices'][0]['message']['content']
                
                # 检查是否包含 <think> 标签，如果有则提取 </think> 之后的内容
                if '</think>' in raw_content:
                    pred_text = raw_content.split('</think>')[-1].strip()
                else:
                    pred_text = raw_content.strip()
        
        # 获取相关信息
        unique_id = data_item.get('unique_id', 'unknown')
        answer = data_item.get('answer', '')
        # answer = data_item.get('original_answer', '')
        random_string_to_prepend = data_item.get('random_string_to_prepend', '')
        category = extract_category_from_unique_id(unique_id)
        
        # 计算得分
        score = evaluate_single_item(pred_text, answer, random_string_to_prepend)
        
        result = {
            'index': index,
            'unique_id': unique_id,
            'category': category,
            'prediction': pred_text,
            'answer': answer,
            'usage': response['usage'],
            'random_string_to_prepend': random_string_to_prepend,
            'score': score,
            'message': message,
            'response': response,
            'timestamp': time.time()
        }
        
        # 写入临时文件（线程安全）
        temp_file = os.path.join(temp_dir, f"result_{index:06d}.json")
        with lock:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"第 {index+1} 个数据项处理完成，结果已写入 {temp_file}")
        return result
        
    except Exception as e:
        print(f"处理第 {index+1} 个数据项时出错: {e}")
        return None

def main(jsonl_file_path, api_url, model_name="glm-4v", headers=None, max_workers=5):
    """主函数：读取JSON文件，使用多线程处理数据并发送到API，并进行评分"""
    try:
        # 读取JSON文件
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f if line.strip()][:]
            # data_list = random.sample(data_list, 150)
        
        if not isinstance(data_list, list):
            print("错误：JSON文件内容不是一个列表")
            return
        
        print(f"共有 {len(data_list)} 个数据项需要处理，使用 {max_workers} 个线程")
        
        # 创建临时目录存储中间结果
        temp_dir = "./mrcr/tmp/"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"临时结果将保存到: {temp_dir}")
        
        # 线程锁，用于文件写入的线程安全
        lock = threading.Lock()
        
        # 使用线程池处理所有数据项
        results = []
        category_results = defaultdict(list)
        
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
                        category_results[result['category']].append(result)
                        
                    print(f"进度: {completed}/{len(data_list)} 完成")
                    
                except Exception as e:
                    print(f"处理第 {index+1} 个数据项时出错: {e}")
        
        # 按index排序结果
        results.sort(key=lambda x: x['index'])
        
        # 按类别计算平均分
        evaluation_results = {}
        total_score = 0
        total_count = 0
        
        print("\n开始计算各类别平均分...")
        for category, category_items in category_results.items():
            if len(category_items) > 0:
                category_scores = [item['score'] for item in category_items]
                avg_score = sum(category_scores) / len(category_scores)
                evaluation_results[category] = {
                    'avg_score': avg_score,
                    'count': len(category_items),
                    'scores': category_scores,
                    'items': category_items
                }
                total_score += sum(category_scores)
                total_count += len(category_scores)
                print(f"类别 {category}: 平均分 {avg_score:.4f}, 样本数 {len(category_items)}")
        
        # 计算总体平均分
        if total_count > 0:
            overall_avg = total_score / total_count
            evaluation_results['overall'] = {
                'avg_score': overall_avg,
                'total_count': total_count
            }
            print(f"\n总体平均分: {overall_avg:.4f}, 总样本数: {total_count}")
        
        print("所有数据处理和评分完成")
        print(f"临时结果文件保存在: {temp_dir}")
        
        return {
            'results': results,
            'evaluation': evaluation_results,
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
    jsonl_file = "./mrcr/data/processed_2needle_0-128k.jsonl"
    # 调用主函数，降低并发数避免超时
    result = main(jsonl_file, URL, model_name="glm-4v", headers=headers, max_workers=10)
    
    if result:
        # 保存详细结果
        output_dir = "./mrcr/results/"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果（包含得分）
        with open(os.path.join(output_dir, "predictions_with_scores.jsonl"), 'w', encoding='utf-8') as f:
            for item in result['results']:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # 保存评分结果
        with open(os.path.join(output_dir, "evaluation_by_category.json"), 'w', encoding='utf-8') as f:
            json.dump(result['evaluation'], f, ensure_ascii=False, indent=2)
        
        # 保存详细的分类结果
        detailed_results = {}
        for category, category_data in result['evaluation'].items():
            if category != 'overall' and 'items' in category_data:
                detailed_results[category] = {
                    'avg_score': category_data['avg_score'],
                    'count': category_data['count'],
                    'individual_scores': [
                        {
                            'unique_id': item['unique_id'],
                            'prediction': item['prediction'],
                            'answer': item['answer'],
                            'score': item['score']
                        } for item in category_data['items']
                    ]
                }
        
        with open(os.path.join(output_dir, "detailed_results_by_category.json"), 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n最终结果已保存到 {output_dir}")
        print(f"预测结果(含分数): {os.path.join(output_dir, 'predictions_with_scores.jsonl')}")
        print(f"按类别评分结果: {os.path.join(output_dir, 'evaluation_by_category.json')}")
        print(f"详细分类结果: {os.path.join(output_dir, 'detailed_results_by_category.json')}")
        print(f"临时结果文件在: {result['temp_dir']}")

