import urllib.request
import json

url = "http://127.0.0.1:11434/api/chat"
payload = {
    "model": "deepseek-r1:8b",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
    "options": {
        "num_ctx": 4096
    }
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data)
req.add_header('Content-Type', 'application/json')

try:
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.getcode()}")
        print(f"Response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
