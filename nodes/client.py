import torch.utils
import torch.utils
from global_var import *


class ClientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, data):
        x = self.conv_stack1(data)
        return x


class Client:
    def __init__(self, port):
        self.port = port
        self.batch_size = 128
        self.epochs = 10
        self.num_nodes = 12
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClientNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.get_data()

    def get_data(self):
        training_dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        data_portion = len(training_dataset) // self.num_nodes
        start_index = (self.port - 8000) * data_portion
        end_index = (self.port - 8000 + 1) * data_portion
        indexes = list(range(start_index, end_index))
        self.training_dataset = torch.utils.data.Subset(training_dataset, indexes)
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.batch_size)

    def load_model(self):
        pass

    def are_models_equal(self, model1, model2):
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()
        
        for key in model1_state_dict:
            if not torch.equal(model1_state_dict[key], model2_state_dict[key]):
                return False
        return True

    def train(self, server_port):
        self.model.train()
        for _ in range(self.epochs):
            for batch, (X, y) in enumerate(self.training_dataloader):
                X = X.to(self.device)
                output = self.model(X)
                clientOutput = output.clone().detach().requires_grad_(True)
                res = requests.post(f"http://localhost:{server_port}/server/train/",
                                    json={
                                        "client_port" : self.port,
                                        "batch": batch,
                                        "clientOutput": json.dumps(clientOutput.tolist()),
                                        "targets": json.dumps(y.tolist())
                                    })
                status = json.loads(res.content.decode())["status"]
                while status == "In progress":
                    time.sleep(0.1)
                    res = requests.post(f"http://localhost:{server_port}/server/tasks/",
                                        json={
                                            "client_port" : self.port
                                        })
                    status = json.loads(res.content.decode())["status"]
                grads = torch.tensor(json.loads(json.loads(res.content.decode())["grads"]))
                output.backward(grads)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        requests.post(f"http://localhost:{server_port}/server/round/",
                                        json={
                                            "client_port" : self.port
                                        })
