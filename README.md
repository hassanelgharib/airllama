# AirLLM API - Ollama-Compatible Wrapper

An Ollama-compatible API wrapper for [AirLLM](https://github.com/lyogavin/airllm), enabling you to run large language models (70B+) on low-memory GPUs (4GB+) with full Ollama client compatibility.

## Features

- 🔌 **Drop-in Ollama replacement** - Works with existing Ollama clients and tools
- 🌐 **Dual API support** - Native Ollama API (`/api/*`) + OpenAI-compatible API (`/v1/*`)
- 💾 **Low memory inference** - Run 70B models on 4GB GPUs using AirLLM's memory optimization
- 🗜️ **Built-in compression** - Supports 4-bit and 8-bit quantization for 3x speed improvement
- 🔄 **Streaming responses** - Real-time token streaming for both APIs
- 🤗 **HuggingFace integration** - Use any HuggingFace model directly

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd airllm-api

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Running the Server

```bash
# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 11434

# Or use the main module
python -m app.main
```

### Using Docker

```bash
# Build the image
docker build -t airllm-api .

# Run the container
docker run -d \
  --name airllm-api \
  --gpus all \
  -p 11434:11434 \
  -v ~/.cache/airllm:/root/.cache/airllm \
  airllm-api
```

## Usage Examples

### Using Ollama Python Client

```python
from ollama import Client

client = Client(host='http://localhost:11434')

# Pull a model (downloads from HuggingFace)
client.pull('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# Generate completion
response = client.generate(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    prompt='Why is the sky blue?'
)
print(response['response'])

# Chat completion
response = client.chat(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[
        {'role': 'user', 'content': 'Hello!'}
    ]
)
print(response['message']['content'])
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required but ignored
)

# Chat completion
response = client.chat.completions.create(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[
        {'role': 'user', 'content': 'Tell me a joke'}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    messages=[{'role': 'user', 'content': 'Count to 10'}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

### Using curl

```bash
# Generate completion
curl http://localhost:11434/api/generate -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat completion
curl http://localhost:11434/api/chat -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}'

# List models
curl http://localhost:11434/api/tags

# Pull a model
curl http://localhost:11434/api/pull -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}'
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `11434` | Server port (Ollama default) |
| `MODEL_CACHE_DIR` | `~/.cache/airllm` | Where to store models |
| `DEFAULT_COMPRESSION` | `4bit` | Compression type: `none`, `4bit`, `8bit` |
| `MAX_LOADED_MODELS` | `1` | Max models in memory (recommend 1 for low-mem) |
| `HF_TOKEN` | - | HuggingFace token for gated models |
| `DEFAULT_MAX_LENGTH` | `2048` | Default context length |
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Default max tokens to generate |

## Supported Endpoints

### Ollama Native API (`/api/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate text completion |
| `/api/chat` | POST | Generate chat completion |
| `/api/embed` | POST | Generate embeddings |
| `/api/tags` | GET | List models |
| `/api/show` | POST | Show model info |
| `/api/pull` | POST | Download model from HuggingFace |
| `/api/copy` | POST | Copy model |
| `/api/delete` | DELETE/POST | Delete model |
| `/api/ps` | GET | List running models |
| `/api/version` | GET | Get API version |

### OpenAI-Compatible API (`/v1/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Embeddings |
| `/v1/models` | GET | List models |
| `/v1/models/{id}` | GET | Get model info |

## Model Names

Unlike Ollama, AirLLM API uses **HuggingFace repository IDs** directly as model names:

- ✅ `meta-llama/Llama-2-7b-hf`
- ✅ `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- ✅ `mistralai/Mistral-7B-Instruct-v0.2`
- ❌ `llama2:7b` (Ollama-style names not supported)

## Supported Models

AirLLM supports most HuggingFace transformer models including:

- **Llama** (Llama 2, Llama 3, etc.)
- **Mistral** / **Mixtral**
- **Qwen** / **Qwen2**
- **ChatGLM**
- **Baichuan**
- **InternLM**

See [AirLLM documentation](https://github.com/lyogavin/airllm#supported-models) for the full list.

## Performance Tips

1. **Use compression** - Set `DEFAULT_COMPRESSION=4bit` for 3x speed improvement with minimal quality loss
2. **Adjust context length** - Smaller `DEFAULT_MAX_LENGTH` uses less memory
3. **Single model** - Keep `MAX_LOADED_MODELS=1` to minimize memory usage
4. **GPU memory** - AirLLM can run 70B models on 4GB VRAM, 405B on 8GB VRAM

## Limitations

- **One model at a time** - Loading a new model unloads the previous one (by design for low-memory operation)
- **Simulated streaming** - AirLLM generates all tokens at once; streaming re-tokenizes the output
- **Basic embeddings** - Uses mean pooling; recommend dedicated embedding models for production
- **No model creation** - `/api/create` returns 501 (use HuggingFace models directly)
- **No model pushing** - `/api/push` returns 501 (local wrapper, not a registry)
- **Chat templates** - Falls back to simple formatting for models without `apply_chat_template`

## Troubleshooting

### Out of disk space during model download
AirLLM splits models into layers which is disk-intensive. Ensure sufficient space and clear HuggingFace cache if needed:
```bash
rm -rf ~/.cache/huggingface/hub
```

### Model fails to load
Check:
1. Model name is a valid HuggingFace repo ID
2. You have a HuggingFace token if the model is gated (set `HF_TOKEN`)
3. Sufficient disk space for model splitting

### Slow generation
Enable compression:
```bash
DEFAULT_COMPRESSION=4bit
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 11434

# View logs
# Logs are printed to stdout in format:
# TIMESTAMP - LOGGER - LEVEL - MESSAGE
```

## License

[Your License Here]

## Acknowledgments

- [AirLLM](https://github.com/lyogavin/airllm) - Low-memory LLM inference
- [Ollama](https://github.com/ollama/ollama) - API design inspiration
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## Contributing

Contributions welcome! Please open an issue or PR.
