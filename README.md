# Airllama

An Ollama-compatible server and CLI for running large language models locally using [AirLLM](https://github.com/lyogavin/airllm) — which lets you run 70B+ models on as little as 4 GB of VRAM through layer-by-layer inference.

## Features

- **`airllama` CLI** — Ollama-style commands: `serve`, `pull`, `list`, `run`, `show`, `ps`, `rm`
- **Ollama-compatible API** — Drop-in replacement at `http://localhost:11434`
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **HuggingFace model IDs** — Use any public or gated HuggingFace model directly
- **CPU-first defaults** — Works on CPU out of the box; enable 4-bit/8-bit quantization when CUDA is available
- **Streaming responses** — Real-time token streaming for both API flavors

---

## Platform Compatibility

| Platform | CPU inference | 4-bit / 8-bit quantization | Notes |
|---|---|---|---|
| Windows 10/11 | ✅ | ✅ (NVIDIA GPU + CUDA) | Use PowerShell; `curl.exe` instead of `curl` in PS 5.1 |
| Linux | ✅ | ✅ (NVIDIA GPU + CUDA) | Recommended platform for GPU inference |
| macOS (Intel) | ✅ | ✅ (NVIDIA eGPU only) | Rare; treat as CPU-only in practice |
| macOS (Apple Silicon) | ✅ | ❌ | No CUDA support — MPS/Metal not supported by bitsandbytes; CPU mode only |

> **`DEFAULT_COMPRESSION` must be left empty on any machine without an NVIDIA CUDA GPU.**

---

## Installation

```bash
git clone https://github.com/hassanelgharib/airllama.git
cd airllama

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Install the airllama CLI
pip install -e .
```

---

## Environment Setup

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell or cmd)
copy .env.example .env
```

Edit `.env` to match your setup:

```env
HOST=0.0.0.0
PORT=11434
MODEL_CACHE_DIR=~/.cache/airllm

# Leave empty for CPU-only.  Set to 4bit or 8bit if you have CUDA + bitsandbytes.
DEFAULT_COMPRESSION=

MAX_LOADED_MODELS=1
HF_TOKEN=
DEFAULT_MAX_LENGTH=2048
DEFAULT_MAX_NEW_TOKENS=512
```

> **All platforms:** Do not put comments on the same line as a value (`VALUE=foo # comment`).
> pydantic-settings will include the comment text in the value. Keep comments on their own lines.
>
> **MODEL_CACHE_DIR** expands `~` correctly on all platforms:
> `C:\Users\<user>\.cache\airllm` on Windows, `/home/<user>/.cache/airllm` on Linux, `/Users/<user>/.cache/airllm` on macOS.

### Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `11434` | Server port (Ollama default) |
| `MODEL_CACHE_DIR` | `~/.cache/airllm` | Model download location |
| `DEFAULT_COMPRESSION` | *(empty)* | `4bit` or `8bit` — requires CUDA + bitsandbytes |
| `MAX_LOADED_MODELS` | `1` | Max models kept in memory simultaneously |
| `HF_TOKEN` | *(empty)* | HuggingFace token for private/gated models |
| `DEFAULT_MAX_LENGTH` | `2048` | Context window length |
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Max tokens to generate per request |

---

## CLI Usage

### Start the server

```bash
airllama serve
```

The server starts on `http://localhost:11434` by default.

### Download a model

```bash
airllama pull TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Model names are HuggingFace repository IDs. Download progress is shown in the terminal.

### List downloaded models

```bash
airllama list
```

### Run a model interactively

```bash
airllama run TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Opens an interactive prompt (type `/bye` to exit).

### Show model info

```bash
airllama show TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### List running models

```bash
airllama ps
```

### Remove a model

```bash
airllama rm TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## API Usage

> **Windows PowerShell 5.1:** `curl` is aliased to `Invoke-WebRequest`. Use `curl.exe` instead for all API examples below. PowerShell 7+ and all Linux/macOS terminals use real curl.

### Ollama native API (`/api/*`)

```bash
# Pull a model
curl http://localhost:11434/api/pull -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

# Generate completion
curl http://localhost:11434/api/generate -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat completion
curl http://localhost:11434/api/chat -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# List models
curl http://localhost:11434/api/tags
```

### OpenAI-compatible API (`/v1/*`)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Tell me a joke"}]
  }'
```

### Python — Ollama client

```python
from ollama import Client

client = Client(host='http://localhost:11434')

# Pull model (if not already downloaded)
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
    messages=[{'role': 'user', 'content': 'What is Python?'}]
)
print(response['message']['content'])
```

### Python — OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='unused')

response = client.chat.completions.create(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[{'role': 'user', 'content': 'Tell me a joke'}]
)
print(response.choices[0].message.content)
```

---

## Supported Endpoints

### Ollama API (`/api/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/generate` | POST | Text completion |
| `/api/chat` | POST | Chat completion |
| `/api/embed` | POST | Embeddings |
| `/api/tags` | GET | List downloaded models |
| `/api/show` | POST | Model info |
| `/api/pull` | POST | Download model from HuggingFace |
| `/api/delete` | DELETE/POST | Remove model |
| `/api/copy` | POST | Copy model |
| `/api/ps` | GET | Running models |
| `/api/version` | GET | API version |

### OpenAI API (`/v1/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Embeddings |
| `/v1/models` | GET | List models |
| `/v1/models/{id}` | GET | Model info |

---

## Supported Models

Any HuggingFace model supported by AirLLM works — including:

- **Llama** (2, 3, 3.1, 3.2)
- **Mistral** / **Mixtral**
- **Qwen** / **Qwen2**
- **Falcon**
- **Baichuan**
- **InternLM**

Use the full HuggingFace repo ID as the model name:

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Llama-3.1-8B-Instruct    <- requires HF_TOKEN
```

> Ollama-style short names like `llama3:8b` are **not** supported — always use the full HuggingFace ID.

---

## Docker

> The default `Dockerfile` uses the `nvidia/cuda:12.1.0-runtime-ubuntu22.04` base image and requires an NVIDIA GPU with CUDA drivers on the host.
> For CPU-only use, change the first line to `FROM python:3.10-slim`.

```bash
# Build
docker build -t airllama .

# Run (CPU-only)
docker run -d \
  --name airllama \
  -p 11434:11434 \
  -v ~/.cache/airllm:/root/.cache/airllm \
  airllama

# Run (NVIDIA GPU)
docker run -d \
  --name airllama \
  --gpus all \
  -p 11434:11434 \
  -v ~/.cache/airllm:/root/.cache/airllm \
  airllama
```

> macOS and Windows Docker Desktop do not expose CUDA GPUs to containers — GPU runs are Linux-only.

---

## Troubleshooting

**`bitsandbytes` error on CPU-only machine**
Leave `DEFAULT_COMPRESSION=` empty in `.env`. 4-bit/8-bit quantization requires CUDA.

**Inline comment in `.env` is read as the value**
`DEFAULT_COMPRESSION= # comment` will set compression to `# comment`.
Always put comments on their own lines.

**Model download stalls with no output**
Check that the server terminal is visible — tqdm progress bars print there, not in the calling client.

**Gated model returns 401**
Set `HF_TOKEN=<your_token>` in `.env` and restart the server.

**macOS Apple Silicon — `bitsandbytes` or CUDA error**
Apple Silicon has no CUDA support. Leave `DEFAULT_COMPRESSION=` empty. Inference runs on CPU.

**Windows PowerShell 5.1 — curl returns HTML or an error**
`curl` in PS 5.1 calls `Invoke-WebRequest`, not the real curl binary. Use `curl.exe` instead:
```powershell
curl.exe http://localhost:11434/api/tags
```
Or upgrade to [PowerShell 7+](https://aka.ms/powershell) where `curl` invokes the real binary.
