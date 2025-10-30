#!/usr/bin/env python3
"""
Simple VLM (Vision Language Model) Inference

Usage:
    from vlm_inference import vlm_inference
    
    response = vlm_inference(
        question="What is in this image?",
        image_paths=["./image1.png", "./image2.png"]
    )
"""

import os
import json
import base64
import io
import math
import requests
from typing import List, Optional
from PIL import Image

# Clear proxy settings
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "glyph"
API_KEY = ""
API_URL = "http://your_api_url:port/v1/chat/completions"
MAX_PIXELS = 36000000

# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_with_max_pixels(image_path: str, max_pixels: int = 1000000) -> str:
    """Encode image to base64 string with pixel limit"""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_inference(
    question: str,
    image_paths: Optional[List[str]] = None,
    api_url: str = API_URL,
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    max_pixels: int = MAX_PIXELS,
    max_tokens: int = 8192,
    temperature: float = 0.0001
) -> Optional[str]:
    """
    Vision Language Model Inference
    
    Args:
        question: Your question/prompt
        image_paths: List of image file paths (optional)
        api_url: API endpoint URL
        api_key: API authentication key
        model_name: Model name
        max_pixels: Maximum pixels for image encoding
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        str: Model response text, or None if error
        
    Example:
        >>> response = vlm_inference(
        ...     question="What is in this image?",
        ...     image_paths=["./image1.png", "./image2.png"]
        ... )
        >>> print(response)
    """
    # Clean input
    question = question.strip() if question else ""
    
    # Build message content
    user_contents = []
    
    # Add images if provided
    if image_paths:
        for image_path in image_paths:
            image_path = image_path.strip()
            if os.path.exists(image_path):
                try:
                    encoded = encode_image_with_max_pixels(image_path, max_pixels=max_pixels)
                    user_contents.append({
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{encoded}"}
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {image_path}: {e}")
            else:
                print(f"Warning: Image not found: {image_path}")
    
    # Add text question
    user_contents.append({'type': 'text', 'text': question})
    
    # Build request payload
    messages = [{'role': 'user', 'content': user_contents}]
    
    payload = {
        # "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "top_k": 1,
        "temperature": temperature,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
        "stop_token_ids": [151329, 151348, 151336],
        "include_stop_str_in_output": True
    }
    
    headers = {
        'Content-Type': 'application/json'
        # 'Authorization': f'Bearer {api_key}'
    }
    
    # Make API request
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=1200)
        
        # Check HTTP status
        if response.status_code != 200:
            print(f"HTTP error: {response.status_code}, Response: {response.text}")
            return None
        
        # Parse JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}, Raw response: {response.text}")
            return None
        
        # Validate response structure
        if 'choices' not in result or not result['choices']:
            print(f"Invalid response structure: {result}")
            return None
        
        if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
            print(f"Missing content in response: {result}")
            return None
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        print("Request timeout")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example 1: Text only
    # response = vlm_inference(
    #     question="What is 2+2?"
    # )
    # print("Text-only response:")
    # print(response)
    # print()
    
    # Example 2: With images
    response = vlm_inference(
        question="Based on the story in the figures, what is the ending of wolf?",
        image_paths=[
            "./output_images/Little_Red_Riding_Hood/page_001.png"
        ]
    )
    print("Image response:")
    print(response)