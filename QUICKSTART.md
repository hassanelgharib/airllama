# Quick Start Guide - Testing the AirLLM API

## Prerequisites
- Python 3.10+ installed
- ~5GB disk space for dependencies
- (Optional) CUDA-capable GPU for faster inference

## Installation Steps

### 1. Install Dependencies

Choose ONE of these options:

#### Option A: Full Installation (Recommended)
```bash
cd \\tana\dev\airllm-api
pip install -r requirements.txt
```

#### Option B: CPU-Only (Faster Install, Slower Inference)
```bash
cd \\tana\dev\airllm-api
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn[standard] pydantic pydantic-settings transformers airllm sse-starlette python-multipart aiofiles
```

#### Option C: Minimal (For API Structure Testing Only)
```bash
cd \\tana\dev\airllm-api
pip install fastapi uvicorn pydantic pydantic-settings sse-starlette python-multipart aiofiles
```
⚠️ This won't work for actual inference, only for testing API structure

### 2. Create Configuration
```bash
# Copy environment template
cp .env.example .env

# Optional: Edit .env
notepad .env
```

**Recommended settings for testing:**
```env
HOST=0.0.0.0
PORT=11434
MODEL_CACHE_DIR=~/.cache/airllm
DEFAULT_COMPRESSION=4bit
MAX_LOADED_MODELS=1
DEFAULT_MAX_NEW_TOKENS=512
```

### 3. Start the Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 11434
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Starting AirLLM API server...
INFO:     Version: 0.1.0
INFO:     Model cache directory: C:\Users\YourUser\.cache\airllm
INFO:     Default compression: 4bit
INFO:     Model manager initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:11434
```

### 4. Test Basic Endpoints

Open a new terminal:

```bash
# Test health
curl http://localhost:11434/health

# Test version
curl http://localhost:11434/api/version

# Test list models (empty initially)
curl http://localhost:11434/api/tags
```

Expected responses:
```json
// /health
{"status":"healthy","version":"0.1.0","models_loaded":0,"models_registered":0}

// /api/version
{"version":"0.1.0"}

// /api/tags
{"models":[]}
```

## Testing with a Small Model

### 1. Pull TinyLlama (600MB, good for testing)

```bash
curl http://localhost:11434/api/pull -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}'
```

This will download and set up the model. First time takes 5-10 minutes.

### 2. Test Generation

```bash
# Simple generation
curl http://localhost:11434/api/generate -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "What is 2+2?",
  "stream": false
}'
```

### 3. Test Chat

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "user", "content": "Tell me a short joke"}
  ],
  "stream": false
}'
```

### 4. Test Streaming

```bash
# Streaming generation (watch tokens appear in real-time)
curl http://localhost:11434/api/generate -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "Count from 1 to 10:",
  "stream": true
}'
```

### 5. Test OpenAI API

```bash
# OpenAI-compatible chat
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## Using Python Clients

### Install Clients
```bash
pip install ollama openai
```

### Ollama Python Client
```python
from ollama import Client

client = Client(host='http://localhost:11434')

# Pull model (if not already pulled)
client.pull('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# Generate
response = client.generate(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    prompt='Explain what AI is in one sentence.'
)
print(response['response'])

# Chat
response = client.chat(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[
        {'role': 'user', 'content': 'What is Python?'}
    ]
)
print(response['message']['content'])

# Streaming
for chunk in client.generate(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    prompt='Count to 20:',
    stream=True
):
    print(chunk['response'], end='', flush=True)
```

### OpenAI Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required but unused
)

# Chat completion
response = client.chat.completions.create(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[
        {'role': 'user', 'content': 'What is machine learning?'}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[{'role': 'user', 'content': 'Write a haiku about coding'}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

## Testing with Docker

### Build Image
```bash
cd \\tana\dev\airllm-api
docker build -t airllm-api .
```

### Run Container
```bash
# CPU-only
docker run -d \
  --name airllm-api \
  -p 11434:11434 \
  -v airllm-cache:/root/.cache/airllm \
  airllm-api

# With GPU
docker run -d \
  --name airllm-api \
  --gpus all \
  -p 11434:11434 \
  -v airllm-cache:/root/.cache/airllm \
  airllm-api
```

### Test
```bash
curl http://localhost:11434/health
```

## Troubleshooting

### Problem: "No module named 'airllm'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Problem: "No module named 'torch'"
**Solution:** Install PyTorch:
```bash
pip install torch transformers airllm
```

### Problem: "CUDA out of memory"
**Solution:** Use CPU or smaller model:
```bash
# Set to CPU in .env
DEFAULT_COMPRESSION=4bit

# Or use CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Model download fails
**Solution:** Check internet connection and disk space:
```bash
# Check space
dir ~\.cache\airllm

# Clear cache if needed
rd /s ~\.cache\airllm
```

### Problem: "Address already in use"
**Solution:** Change port in .env:
```env
PORT=11435
```

### Problem: Slow generation
**Solution:** Enable compression:
```env
DEFAULT_COMPRESSION=4bit  # 3x faster with minimal quality loss
```

## Recommended Test Models

| Model | Size | Best For | RAM Required |
|-------|------|----------|--------------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 600MB | Testing | 2GB |
| microsoft/phi-2 | 2.7GB | Small tasks | 4GB |
| mistralai/Mistral-7B-Instruct-v0.2 | 7GB | General use | 8GB |
| meta-llama/Llama-2-7b-chat-hf | 13GB | Chat | 12GB |

## Automated Test Script

Run the included test script:
```bash
python test_server.py
```

This will:
1. Start server in background
2. Test all basic endpoints
3. Report results
4. Shutdown server

## Next Steps

1. ✅ Verify server starts
2. ✅ Test basic endpoints
3. ✅ Pull a small model
4. ✅ Test generation
5. ✅ Test chat
6. ✅ Test streaming
7. ✅ Test with Python clients
8. ⚠️ Test with your own models
9. ⚠️ Test with production workload
10. ⚠️ Add monitoring and logging

## Support

- GitHub Issues: [Your Repo URL]
- AirLLM Docs: https://github.com/lyogavin/airllm
- Ollama API Docs: https://github.com/ollama/ollama/blob/main/docs/api.md

---

**Happy Testing! 🚀**
