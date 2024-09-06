from global_var import *


executer = concurrent.futures.ThreadPoolExecutor(3)
losses = []
app = Flask(__name__)

# def get_data(num_nodes):
#     test_file = open(f"./data/femnist/test/node{num_nodes}.json", "r")
#     test_data = json.loads(test_file.read())
#     test_dataset = CustomImageDataset(test_data)
#     return DataLoader(test_dataset, batch_size=len(test_dataset))

def get_data(num_nodes):
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )

    test_portion = len(test_dataset) // 9
    test_start_index = 0
    test_end_index = test_portion
    test_indexes = list(range(test_start_index, test_end_index))

    test_dataset = torch.utils.data.Subset(test_dataset, test_indexes)
    return DataLoader(test_dataset, batch_size=test_portion)


def evaluate(num_nodes):
    test_dataloader = get_data(num_nodes)
    client_model = ClientNN().to("cpu")
    client_model.load_state_dict(torch.load("./models/global_client.pth"))
    server_model = ServerNN().to("cpu")
    server_model.load_state_dict(torch.load("./models/global_server.pth"))
    loss_fn = nn.CrossEntropyLoss()
    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to("cpu")
            outputs = server_model(client_model(X))
            loss = loss_fn(outputs, y)
            losses.append(loss.item())


def aggregate_models(models):
    global_model = {}
    for key in models[0].keys():
        global_model[key] = sum(model[key] for model in models) / len(models)
    return global_model


def aggregate_splitfed(server_names, client_names):
    servers = [torch.load(f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/blockchain_split_fed/models/{server_name}_server.pth")
               for server_name in server_names]
    clients = [torch.load(f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/blockchain_split_fed/models/{client_name}_client.pth")
               for client_name in client_names]
    global_server = aggregate_models(servers)
    global_client = aggregate_models(clients)
    torch.save(global_server, "./models/global_server.pth")
    torch.save(global_client, "./models/global_client.pth")
    return "Aggregation completed."

@app.route("/losses/")
def save_losses():
    file = open(f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/blockchain_split_fed/losses/aggregator.json", "w")
    file.write(json.dumps(losses))
    return "done"


@app.route("/aggregate/", methods=['POST'])
def aggregate():
    servers = request.get_json()["servers"]
    clients = request.get_json()["clients"]
    num_nodes = request.get_json()["numNodes"]
    msg = aggregate_splitfed(servers, clients)
    evaluate(num_nodes)
    return msg


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)
