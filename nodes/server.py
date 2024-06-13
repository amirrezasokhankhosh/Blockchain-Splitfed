from global_var import *
import torch.nn.functional as F

class ServerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2))
        )

        self.conv_stack3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2))
        )

        self.classification_stack = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(4*4*128, 10),
            nn.Softmax(1)
        )

    def forward(self, data):
        x = self.conv_stack2(data)
        x = self.conv_stack3(x)
        return self.classification_stack(x)


class Server:
    def __init__(self, port):
        self.port = port
        self.rounds = 2
        self.current_round = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 3e-4
        self.avg_model = ServerNN().to(self.device)
        self.models = {}
        self.losses = {}
        self.round_completion = {}
        self.semaphore = threading.Semaphore(1)

    def load_model(self):
        # TODO: Load the previous global model
        pass

    def get_model(self, client_port):
        with self.semaphore:
            return self.models[client_port]

    def set_model(self, model, client_port):
        with self.semaphore:
            self.models[client_port].load_state_dict(model.state_dict())

    def train(self, client_port, batch, clientOutputCPU, targets):
        model = self.get_model(client_port)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam(self.avg_model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.to(self.device)
        # self.avg_model.to(self.device)
        targets = targets.to(self.device)
        # model.train()
        # self.avg_model.train()
        optimizer.zero_grad()
        clientOutputCPU = torch.tensor(clientOutputCPU).requires_grad_(True)
        clientOutput = clientOutputCPU.to(self.device)
        pred = model(clientOutput)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        self.set_model(model, client_port)
        self.losses[client_port][self.current_round].append(loss.item())
        return clientOutputCPU.grad.clone().detach()


    def aggregate(self):
        clients = list(self.models.keys())
        weights_avg = copy.deepcopy(self.models[clients[0]].state_dict())
        for k in weights_avg.keys():
            for i in range(1, len(clients)):
                weights_avg[k] += self.models[clients[i]].state_dict()[k]
            weights_avg[k] = torch.div(weights_avg[k], len(clients))
        self.avg_model.load_state_dict(weights_avg)


    def finish_round(self, client):
        self.round_completion[client] = True
        for client, completed in self.round_completion.items():
            if not completed: 
                return
        self.aggregate()
        if self.current_round != self.rounds:
            self.start_round()
        else:
            self.save_losses()
            print("Training Completed.")
        


    def start(self, clients):
        self.losses = {}
        self.clients = clients
        for client in self.clients:
            self.losses[client] = {}
        
        self.start_round()

    
    def start_round(self):
        self.current_round += 1
        self.models = {}
        # TODO: self.avg_model = self.load_model()
        for client in self.clients:
            self.models[client] = copy.deepcopy(self.avg_model)
            self.losses[client][self.current_round] = []
            self.round_completion[client] = False

        for client in self.clients:
            requests.get(f"http://localhost:{client}/client/train/",
                         json={
                             "server_port": self.port
                         })
    
    def save_losses(self):
        file = open("losses.json", "w")
        file.write(json.dumps(self.losses))

