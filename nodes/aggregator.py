from global_var import *


executer = concurrent.futures.ThreadPoolExecutor(2)
app = Flask(__name__)


def aggregate_models(models):
    global_model = {}
    for key in models[0].keys():
        global_model[key] = sum(model[key] for model in models) / len(models)
    return global_model


def aggregate_splitfed(server_names, client_names):
    servers = [torch.load(f"./models/{server_name}_server.pth")
               for server_name in server_names]
    clients = [torch.load(f"./models/{client_name}_client.pth")
               for client_name in client_names]
    global_server = aggregate_models(servers)
    global_client = aggregate_models(clients)
    torch.save(global_server, "./models/global_server.pth")
    torch.save(global_client, "./models/global_client.pth")
    return "Aggregation completed."


@app.route("/aggregate/", methods=['POST'])
def aggregate():
    servers = request.get_json()["servers"]
    clients = request.get_json()["clients"]
    return aggregate_splitfed(servers, clients)
    # executer.submit(aggregate_weights, models)
    # return "aggregation started."


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)
