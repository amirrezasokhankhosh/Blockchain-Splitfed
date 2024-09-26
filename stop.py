import requests

# nodes
for i in range(9):
    try:
        requests.get(f"http://localhost:{8000 + i}/exit/")
    except:
        print(f"Node{i} is stopped.")

try:
    requests.get(f"http://localhost:{5050}/exit/")
except:
    print(f"Fed server is stopped.")