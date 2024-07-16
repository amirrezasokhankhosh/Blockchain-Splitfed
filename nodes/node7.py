from global_var import *
from client import Client
from server import Server


port = 8007
num_clients = 3
futures = {}
executer = concurrent.futures.ThreadPoolExecutor(num_clients+1)
client = Client(port)
server = Server(port)
app = Flask(__name__)


@app.route("/client/train/")
def train_client():
    server_port = request.get_json()["server_port"]
    # client.train(server_port)
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

    return {
        "status" : "In progress"
    }


@app.route("/server/tasks/", methods=['POST'])
def check_tasks():
    client_port = request.get_json()["client_port"]
    future = futures[client_port]
    if not future.done():
        return {
            "status" : "In progress"
        }   

    grads, loss = future.result()
    del futures[client_port]

    return {
        "status" : "Completed",
        "grads" : json.dumps(grads.tolist()),
        "loss" : loss
    }


@app.route("/server/round/", methods=['POST'])
def round_completed():
    client_port = request.get_json()["client_port"]
    losses = request.get_json()["losses"]
    server.finish_round(client_port, losses)
    return "Well Done."

@app.route("/server/models/ready/")
def models_ready():
    # executer.submit(server.evaluate, client)
    try:
        server.evaluate(client)
    except Exception as e:
        print(e)
        raise Exception("error")
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
