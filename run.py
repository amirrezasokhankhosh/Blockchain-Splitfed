import os
import time
import subprocess
import requests


def create_node(i, num_clients=1):
    code = f"""from global_var import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sharding_split_fed.client import Client
from sharding_split_fed.server import Server


port = {8000 + i}
num_clients = {num_clients}
futures = {{}}
executer = concurrent.futures.ThreadPoolExecutor(num_clients+2)
client = Client(port, ClientNN)
server = Server(port, ServerNN)
app = Flask(__name__)


@app.route("/client/train/")
def train_client():
    server_port = request.get_json()["server_port"]
    executer.submit(client.train, server_port)
    return "The client started training!"


@app.route("/client/load/")
def load_client():
    client.load_model()
    return "The client model is loaded."


@app.route("/server/train/", methods=['POST'])
def train_server():
    client_port = request.get_json()["client_port"]
    batch = request.get_json()["batch"]
    clientOutputCPU = json.loads(request.get_json()["clientOutput"])
    targets = torch.tensor(json.loads(request.get_json()["targets"]))

    future = executer.submit(server.train, client_port, batch, clientOutputCPU, targets)
    futures[client_port] = future

    return {{
        "status" : "In progress"
    }}


@app.route("/server/tasks/", methods=['POST'])
def check_tasks():
    client_port = request.get_json()["client_port"]
    future = futures[client_port]
    if not future.done():
        return {{
            "status" : "In progress"
        }}   

    grads, loss = future.result()
    del futures[client_port]

    return {{
        "status" : "Completed",
        "grads" : json.dumps(grads.tolist()),
        "loss" : loss
    }}


@app.route("/server/round/", methods=['POST'])
def round_completed():
    client_port = request.get_json()["client_port"]
    losses = request.get_json()["losses"]
    server.finish_round(client_port, losses)
    return "Well Done."

@app.route("/server/models/ready/")
def models_ready():
    executer.submit(server.evaluate, client)
    return "done"

@app.route("/server/", methods=['POST'])
def start_server():
    temp = request.get_json()["clients"]
    clients = [c["port"] for c in temp]
    cycle = request.get_json()["cycle"]
    executer.submit(server.start, clients, cycle)
    return "Started."

@app.route("/save/")
def plot():
    server.save_losses()
    return "Done"

@app.route("/exit/")
def exit_miner():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port, debug=True)
"""
    file = open(f"./node{i}.py", "w")
    file.write(code)
    file.close()


def run_node(i):
    # Run node{i}.py in the background and redirect output to a log file
    log_file = f"../logs/node_{i}.txt"
    with open(log_file, "w") as f:
        subprocess.Popen(
            ["python3", f"./node{i}.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

if __name__ == "__main__":
    num_nodes = 36
    num_clients = 5
    cwd = os.path.dirname(__file__)

    os.chdir(os.path.join(cwd, "nodes"))
    print("Create Nodes.")
    for i in range(num_nodes):
        create_node(i, num_clients=num_clients)
    
    # Step 1: Bring up miners (start processes)
    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        run_node(i)
        time.sleep(1)
    
    with open("../logs/fed_server.txt", "w") as f:
        subprocess.Popen(
            ["python3", f"./fed_server.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

    # Step 7: Re-initializing the model
    print("Re-initializing the global model...")
    os.system("python3 ./model.py")
    time.sleep(1)

    print("Starting the mining process...")
    requests.get("http://localhost:5050/start/")

    # The main script ends here, while the node processes continue running in the background
    print("Nodes have been started, and the script has completed.")