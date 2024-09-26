import requests

# Express Apps
try:
    requests.get("http://localhost:3000/exit/")
except:
    print("App1 is stopped.")

# Nodes
for i in range(9):
    try:
        requests.get(f"http://localhost:{8000 + i}/exit/")
    except:
        print(f"Node{i} is stopped.")

# Aggregator
try:
    requests.get("http://localhost:5050/exit/")
except:
    print("Aggregator is stopped.")
