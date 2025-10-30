import re
import json
import os
import sys
from pathlib import Path
import shutil

# def extract_answer_from_text(text):
#     """
#     从包含 <think> 标签的文本中提取最终答案
#     """
#     if not text:
#         return text
    
#     # 移除 <think> 标签及其内容
#     # 匹配 <think> 到 </think> 或者到下一个标签的内容
#     text = re.sub(r'<think>.*?(?:</think>|(?=<\w+>)|$)', '', text, flags=re.DOTALL)
    
#     # 移除其他可能的标签如 <|user|>, <|assistant|> 等
#     text = re.sub(r'<\|[^|]+\|>', '', text)
    
#     # 移除多余的空白字符和换行符
#     text = re.sub(r'\n+', ' ', text)
#     text = text.strip()
    
#     return text

def extract_answer_from_text(content):
    import re
    # 移除 <think> 标签及其内容
    pattern = r"<think>.*?</think>\s*"
    result = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # 移除其他标签如 <|user|>, <|assistant|> 等
    result = re.sub(r'<\|[^|]+\|>', '', result)
    
    # 移除换行符和多余空白
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

def clean_jsonl_file(input_file_path, output_file_path):
    """
    清理 JSONL 文件（每行一个 JSON 对象）
    """
    try:
        cleaned_lines = []
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    cleaned_lines.append(line)
                    continue
                
                try:
                    # 解析每行的 JSON
                    data = json.loads(line)
                    
                    # 清理 pred 字段
                    if isinstance(data, dict) and 'pred' in data:
                        data['pred'] = extract_answer_from_text(data['pred'])
                    
                    # 转换回 JSON 字符串
                    cleaned_line = json.dumps(data, ensure_ascii=False)
                    cleaned_lines.append(cleaned_line)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {line_num} 行 JSON 解析错误: {e}")
                    # 如果解析失败，保留原行
                    cleaned_lines.append(line)
        
        # 确保输出目录存在
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"✅ 已清理 JSONL 文件: {input_file_path} -> {output_file_path}")
        
    except Exception as e:
        print(f"❌ 处理 JSONL 文件 {input_file_path} 时出错: {e}")

def clean_json_file(input_file_path, output_file_path):
    """
    清理单个JSON文件并保存到指定位置
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是列表格式
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'pred' in item:
                    item['pred'] = extract_answer_from_text(item['pred'])
        
        # 如果是字典格式
        elif isinstance(data, dict):
            if 'pred' in data:
                data['pred'] = extract_answer_from_text(data['pred'])
            # 处理嵌套结构
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'pred' in item:
                            item['pred'] = extract_answer_from_text(item['pred'])
        
        # 确保输出目录存在
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已清理 JSON 文件: {input_file_path} -> {output_file_path}")
        
    except Exception as e:
        print(f"❌ 处理 JSON 文件 {input_file_path} 时出错: {e}")

def clean_text_file(input_file_path, output_file_path):
    """
    清理普通文本文件并保存到指定位置
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按行处理
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if '"pred":' in line:
                # 提取pred字段的值 - 处理可能包含转义字符的情况
                match = re.search(r'"pred":\s*"([^"\\]*(?:\\.[^"\\]*)*)"', line)
                if match:
                    pred_value = match.group(1)
                    # 解码转义字符
                    try:
                        decoded_value = pred_value.encode().decode('unicode_escape')
                        cleaned_value = extract_answer_from_text(decoded_value)
                        # 重新编码转义字符
                        escaped_value = cleaned_value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        line = line.replace(pred_value, escaped_value)
                    except:
                        # 如果解码失败，直接处理原始值
                        cleaned_value = extract_answer_from_text(pred_value)
                        line = line.replace(pred_value, cleaned_value)
            
            cleaned_lines.append(line)
        
        # 确保输出目录存在
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"✅ 已清理文件: {input_file_path} -> {output_file_path}")
        
    except Exception as e:
        print(f"❌ 处理文件 {input_file_path} 时出错: {e}")

def copy_other_files(input_file_path, output_file_path):
    """
    复制不需要清理的文件到输出目录
    """
    try:
        # 确保输出目录存在
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        shutil.copy2(input_file_path, output_file_path)
        print(f"📋 已复制文件: {input_file_path} -> {output_file_path}")
        
    except Exception as e:
        print(f"❌ 复制文件 {input_file_path} 时出错: {e}")

def main(input_dir=None, output_dir=None):
    """
    主函数：扫描输入目录并将清理后的文件保存到输出目录
    """
    # 如果没有提供参数，从命令行获取
    if input_dir is None:
        if len(sys.argv) < 3:
            print("使用方法: python clear.py <输入文件夹> <输出文件夹>")
            print("例如: python clear.py ./input_data ./cleaned_data")
            return
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_path}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 开始扫描目录: {input_path}")
    print(f"📁 输出目录: {output_path}")
    
    # 递归查找文件
    for file_path in input_path.rglob('*'):
        if file_path.is_file():
            # 计算相对路径
            relative_path = file_path.relative_to(input_path)
            output_file_path = output_path / relative_path
            
            suffix = file_path.suffix.lower()
            
            if suffix == '.jsonl':
                # 专门处理 JSONL 文件
                clean_jsonl_file(file_path, output_file_path)
            elif suffix == '.json':
                # 处理普通 JSON 文件
                clean_json_file(file_path, output_file_path)
            elif suffix in ['.txt', '.log']:
                # 处理文本文件
                clean_text_file(file_path, output_file_path)
            else:
                # 复制其他文件
                copy_other_files(file_path, output_file_path)
    
    print("🎉 清理完成！")

if __name__ == "__main__":
    # 测试单个字符串的清理效果
    test_text = ' <think> xxxxx\n <think>  \nNo <|user|>'
    print(f"测试输入: {repr(test_text)}")
    print(f"清理结果: {repr(extract_answer_from_text(test_text))}")
    print("-" * 50)
    
    # 执行主清理程序
    main()