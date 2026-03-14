import requests
import json

url = "http://localhost:8999/"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Augment/1.0"
}
data = {
    "action": "status"
}

response = requests.post(url, headers=headers, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
