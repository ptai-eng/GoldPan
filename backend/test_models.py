import requests

models_to_test = [
    "literally-fake-model-123",
]

for m in models_to_test:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:embedContent?key=dummy_key"
    payload = {
        "model": f"models/{m}",
        "content": {
            "parts": [{"text": "hello"}]
        }
    }
    res = requests.post(url, json=payload)
    print(f"Testing {m}: Status {res.status_code}, Body: {res.text}")
