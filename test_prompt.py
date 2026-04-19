import requests
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'prompt': 'What is 2+2?',
        'stream': False
    }
)
print(response.json())