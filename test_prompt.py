import requests
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'prompt': 'What is 2+2?',
        'stream': False
    }
)
print(response.json())