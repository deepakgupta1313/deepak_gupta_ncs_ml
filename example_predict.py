import requests

host = "http://localhost:5000"

payload = [{"first_name": "Deepak"},
           {"first_name": "Anita"}]

r = requests.post(f"{host}/predict", json=payload)
print(r.text)
