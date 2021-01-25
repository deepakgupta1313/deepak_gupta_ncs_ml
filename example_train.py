import requests

host = "http://localhost:5000"
r = requests.get(f"{host}/train")
print(r.text)
