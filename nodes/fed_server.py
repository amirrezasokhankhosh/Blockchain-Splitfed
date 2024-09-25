from global_var import *


class FedServer:
    def __init__(self):
        self.cycles = 20
        self.num_servers = 3
        self.num_clients = 2
        self.current_cycle = 0
        self.cyle_completion = {}
        self.losses = []
        self.assign_nodes()

    def assign_nodes(self):
        assigned = {"servers": [], "clients": {}}
        for i in range(self.num_servers):
            current_server = {"id": f"node_{i * (self.num_clients + 1)}",
                              "port": 8000 + i * (self.num_clients + 1)}
            assigned["servers"].append(current_server)
            assigned["clients"][current_server["id"]] = []
            for j in range(1, self.num_clients + 1):
                current_client = {"id": f"node_{i * (self.num_clients + 1) + j}",
                                  "port": 8000 + i * (self.num_clients + 1) + j}
                assigned["clients"][current_server["id"]].append(current_client)
        self.assigned = assigned

    def start_cycle(self):
        self.cyle_completion = {}
        self.current_cycle += 1
        if self.current_cycle < self.cycles:
            for server in self.assigned["servers"]:
                self.cyle_completion[server["port"]] = False
                requests.post(f"http://localhost:{server["port"]}/server/",
                              json={
                    "clients": self.assigned["clients"][server["id"]],
                    "cycle": self.current_cycle
                })
        print(f"*** CYCLE {self.current_cycle} STARTED ***")
        print("Training started.")

    def finish_cycle(self, server_port):
        self.cyle_completion[server_port] = True
        for server_port, completed in self.cyle_completion.items():
            if not completed:
                return
        print(f"*** CYCLE {self.current_cycle} COMPLETED ***")
        self.aggregate()
        self.evaluate(num_nodes=0)
        if self.current_cycle != self.cycles:
            self.start_cycle()
        else:
            self.save_losses()
            print("All cycles completed.")

    def aggregate_models(self, models):
        global_model = {}
        for key in models[0].keys():
            global_model[key] = sum(model[key]
                                    for model in models) / len(models)
        return global_model

    def aggregate(self):
        servers = [torch.load(f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/multi_split_fed/models/{server["id"]}_server.pth")
                   for server in self.assigned["servers"]]

        clients = []
        for server in self.assigned["servers"]:
            for client in self.assigned["clients"][server["id"]]:
                clients.append(torch.load(
                    f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/multi_split_fed/models/{client["id"]}_client.pth"))

        global_server = self.aggregate_models(servers)
        global_client = self.aggregate_models(clients)
        torch.save(global_server, "./models/global_server.pth")
        torch.save(global_client, "./models/global_client.pth")
        return "Aggregation completed."

    def get_data(self, num_nodes):
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

    def evaluate(self, num_nodes):
        test_dataloader = self.get_data(num_nodes)
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
                self.losses.append(loss.item())

    def save_losses(self):
        file = open(
            f"/Users/amirrezasokhankhosh/Documents/Workstation/splitfed/multi_split_fed/losses/aggregator.json", "w")
        file.write(json.dumps(self.losses))
        return "done"


executer = concurrent.futures.ThreadPoolExecutor(3)
fed_server = FedServer()
app = Flask(__name__)

@app.route("/start/")
def plot():
    fed_server.start_cycle()
    return "Cycles started."


@app.route("/server/cycle/", methods=['POST'])
def round_completed():
    server_port = request.get_json()["server_port"]
    fed_server.finish_cycle(server_port)
    return "Well Done."


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)
