# Airllama — Quick Start

## Prerequisites

- Python 3.10+
- ~5 GB disk space for dependencies + model cache
- (Optional) NVIDIA GPU with CUDA for 4-bit/8-bit quantization
  - **macOS Apple Silicon**: No CUDA support — CPU mode only (`DEFAULT_COMPRESSION=` empty)
  - **Windows / Linux**: 4-bit/8-bit available with NVIDIA GPU + bitsandbytes

> **All curl examples below use the real `curl` binary.**
> On Windows PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`. Use `curl.exe` explicitly, or upgrade to [PowerShell 7+](https://aka.ms/powershell).

---

## 1. Install

```bash
# Clone
git clone https://github.com/hassanelgharib/airllama.git
cd airllama

# Create venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies + CLI
pip install -r requirements.txt
pip install -e .
```

> **NVIDIA Jetson (JetPack 6):** Install PyTorch from the Jetson AI Lab wheel index *first*, before the line above:
> ```bash
> pip install torch==2.8.0 torchvision==0.23.0 \
>   --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
> ```
> Standard PyPI torch wheels are x86-64 only and will not work on Jetson ARM64.

Verify the CLI is available:

```bash
airllama --help
```

---

## 2. Configure the environment

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell or cmd)
copy .env.example .env
```

Open `.env` and verify these key settings:

```env
HOST=0.0.0.0
PORT=11434
MODEL_CACHE_DIR=~/.cache/airllm

# Leave empty for CPU-only.
# Set to 4bit or 8bit only when you have CUDA + bitsandbytes installed.
DEFAULT_COMPRESSION=

MAX_LOADED_MODELS=1
HF_TOKEN=                 # Required for gated models (Llama 3, etc.)
DEFAULT_MAX_LENGTH=2048
DEFAULT_MAX_NEW_TOKENS=512
```

> **`MODEL_CACHE_DIR`** expands `~` correctly on all platforms:
> - Windows: `C:\Users\<user>\.cache\airllm`
> - Linux: `/home/<user>/.cache/airllm`
> - macOS: `/Users/<user>/.cache/airllm`

> **All platforms:** Never put a comment on the same line as a value.
> `DEFAULT_COMPRESSION= # comment` will cause pydantic-settings to read `# comment` as the value.

---

## 3. Start the server

```bash
airllama serve
```

Expected output:

```
INFO:     Starting Airllama server...
INFO:     Version: 0.1.0
# Windows:  Model cache: C:\Users\<user>\.cache\airllm
# Linux:    Model cache: /home/<user>/.cache/airllm
# macOS:    Model cache: /Users/<user>/.cache/airllm
INFO:     Default compression: None
INFO:     Uvicorn running on http://0.0.0.0:11434
```

---

## 4. Pull a model

In a second terminal (with the venv activated):

```bash
airllama pull mistralai/Mistral-7B-Instruct-v0.2
```

This downloads ~14 GB of model shards. Download progress is shown in the **server** terminal.
The pull command prints status updates and confirms when complete.

> **Why Mistral 7B?** AirLLM requires models stored as multiple shards (typically 7B+ parameters).
> Smaller single-file models like TinyLlama 1.1B are **not compatible** with AirLLM.

For gated models (e.g. Llama 3), set `HF_TOKEN=` in `.env` first.

---

## 5. Verify the model is registered

```bash
airllama list
```

---

## 6. Test via curl

> **Windows PowerShell 5.1:** Replace `curl` with `curl.exe` in every command below.
> PowerShell 7+ and all Linux/macOS terminals work as shown.

```bash
# Health check
curl http://localhost:11434/health

# List models
curl http://localhost:11434/api/tags

# Generate (non-streaming)
curl http://localhost:11434/api/generate -d '{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "prompt": "What is 2+2?",
  "stream": false
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [{"role": "user", "content": "Tell me a short joke"}],
  "stream": false
}'

# Streaming generation
curl http://localhost:11434/api/generate -d '{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "prompt": "Count from 1 to 5:",
  "stream": true
}'

# OpenAI-compatible
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 7. Run the included test script

```bash
python test_prompt.py
```

This sends a single non-streaming request to `/api/generate` and prints the JSON response.

---

## 8. Interactive mode

```bash
airllama run mistralai/Mistral-7B-Instruct-v0.2
```

Type your prompt and press Enter. Type `/bye` to exit.

---

## 9. Python clients

```bash
pip install ollama openai
```

### Ollama

```python
from ollama import Client

client = Client(host='http://localhost:11434')
client.pull('mistralai/Mistral-7B-Instruct-v0.2')

response = client.generate(
    model='mistralai/Mistral-7B-Instruct-v0.2',
    prompt='Explain what AI is in one sentence.'
)
print(response['response'])
```

### OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='unused')

response = client.chat.completions.create(
    model='mistralai/Mistral-7B-Instruct-v0.2',
    messages=[{'role': 'user', 'content': 'What is Python?'}]
)
print(response.choices[0].message.content)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `bitsandbytes` / CUDA error on CPU | Set `DEFAULT_COMPRESSION=` (empty) in `.env` |
| macOS Apple Silicon — CUDA not found | No CUDA on Apple Silicon; leave `DEFAULT_COMPRESSION=` empty |
| Pydantic reads comment as value | Move `.env` comments to their own lines |
| Pull shows no progress | Watch the **server** terminal, not the client |
| 401 on gated model | Set `HF_TOKEN=<token>` in `.env`, restart server |
| Port already in use | Change `PORT=` in `.env` or stop the other process |
| Windows PS 5.1 curl returns HTML | Use `curl.exe` instead of `curl` in PowerShell 5.1 |

