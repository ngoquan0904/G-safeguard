# Fix lỗi 404 - Alibaba Qwen API Configuration

## 🔴 Vấn đề

Khi chạy với Alibaba Qwen API, nhận lỗi:
```
Error code: 404 - 'No static resource v1/chat/completions/chat/completions.'
```

## 🔍 Nguyên nhân

OpenAI client tự động thêm đường dẫn `/chat/completions` vào cuối BASE_URL, dẫn đến:

```
BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
            + "/chat/completions" (tự động thêm)
            = "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions/chat/completions" ❌
```

---

## ✅ Giải pháp 1: Sửa Environment Variable (KHUYÊN DÙNG)

Đơn giản nhất - chỉ cần sửa BASE_URL:

```bash
# ❌ SAI (gây lỗi 404)
export BASE_URL="https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
export OPENAI_API_KEY="sk-sp-0a5e722551b845fb999cdb51e0b1833b"

# ✅ ĐÚNG (loại bỏ /chat/completions khỏi base URL)
export BASE_URL="https://coding-intl.dashscope.aliyuncs.com/v1"
export OPENAI_API_KEY="sk-sp-0a5e722551b845fb999cdb51e0b1833b"
```

**Chạy lại:**
```bash
cd /home/admin123/Documents/VNPT_AI/GuardRAG/G-safeguard/MA

# Cấu hình environment
export BASE_URL="https://coding-intl.dashscope.aliyuncs.com/v1"
export OPENAI_API_KEY="sk-sp-0a5e722551b845fb999cdb51e0b1833b"

# Chạy
chmod +x ./scripts/train/gen_conversation_train.sh
./scripts/train/gen_conversation_train.sh && python merge_datasets.py --phase train
```

---

## ✅ Giải pháp 2: Sửa Code `agents.py` (Tuỳ chọn)

Nếu muốn kiểm soát tốt hơn, có thể thay đổi code để sử dụng requests trực tiếp:

**Thay đổi file:** `agents.py`

```python
import os
import asyncio
import numpy as np
import re
import requests
import aiohttp
from typing import Literal


def llm_invoke(prompt, model_type: str):
    """Call Alibaba Qwen API using requests"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/v1")
    
    # Ensure base_url doesn't end with /chat/completions
    if base_url.endswith("/chat/completions"):
        base_url = base_url.rsplit("/chat/completions", 1)[0]
    
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_type,
        "messages": prompt,
        "temperature": 0,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API Error: {e}")
        raise


async def allm_invoke(prompt, model_type: str):
    """Call Alibaba Qwen API asynchronously using aiohttp"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/v1")
    
    # Ensure base_url doesn't end with /chat/completions
    if base_url.endswith("/chat/completions"):
        base_url = base_url.rsplit("/chat/completions", 1)[0]
    
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_type,
        "messages": prompt,
        "temperature": 0,
        "max_tokens": 1024
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API Error: {e}")
        raise
```

---

## 🧪 Test lại

### Test 1: Kiểm tra API endpoint

```bash
python3 << 'EOF'
import os
import requests

api_key = "sk-sp-0a5e722551b845fb999cdb51e0b1833b"
base_url = "https://coding-intl.dashscope.aliyuncs.com/v1"

url = f"{base_url}/chat/completions"
print(f"Testing API endpoint: {url}")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "qwen-plus",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ],
    "temperature": 0,
    "max_tokens": 100
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ API works! Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")
EOF
```

### Test 2: Chạy data generation

```bash
export BASE_URL="https://coding-intl.dashscope.aliyuncs.com/v1"
export OPENAI_API_KEY="sk-sp-0a5e722551b845fb999cdb51e0b1833b"

cd /home/admin123/Documents/VNPT_AI/GuardRAG/G-safeguard/MA

# Test với 1 command đơn giản (--samples 2 thay vì 40)
python3 gen_graph.py \
  --num_nodes 8 \
  --sparsity 0.2 \
  --num_graphs 2 \
  --num_attackers 1 \
  --samples 2 \
  --model_type qwen-plus \
  --phase test
```

---

## 📋 Tóm tắt cách fix

| Bước | Thao tác |
|------|---------|
| 1 | Set `BASE_URL="https://coding-intl.dashscope.aliyuncs.com/v1"` (không có `/chat/completions`) |
| 2 | Set API key: `OPENAI_API_KEY="sk-sp-0a5e722551b845fb999cdb51e0b1833b"` |
| 3 | Test API endpoint với script kiểm tra |
| 4 | Chạy data generation lại |

---

## 🔧 Ghi chú

1. **Giải pháp 1 (môi trường)** là cách nhanh nhất, không cần sửa code
2. **Giải pháp 2 (code)** cho phép kiểm soát tốt hơn nhưng cần sửa file
3. Model type có thể dùng: `qwen-plus`, `qwen-turbo`, `qwen-max` tùy theo quota
4. Nếu vẫn lỗi, kiểm tra:
   - API key còn hiệu lực không
   - Base URL đúng không
   - Network connection có bình thường không

---

**Generated:** 24/03/2026
