import torch.utils
import torch.utils
from global_var import *


# class ClientNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_stack1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(3, 3), padding="same"),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same"),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d((2, 2))
#         )

#     def forward(self, data):
#         x = self.conv_stack1(data)
#         return x

class ClientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

    def forward(self, data):
        x = self.conv_stack1(data)
        return x
    

class CustomImageDataset(Dataset):
    def __init__(self, data):
        x = torch.Tensor(np.array(data['x']).reshape((-1, 28, 28)))
        resize_transform = transforms.Resize((224, 224))
        self.x = torch.stack([resize_transform(img.unsqueeze(0)) for img in x])
        self.y = torch.Tensor(np.array(data['y']))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx].long()
        return data, label


class Client:
    def __init__(self, port, malicious=False):
        self.port = port
        self.batch_size = 128
        self.epochs = 1
        self.num_nodes = 9
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = ClientNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.get_data()
        self.malicious = malicious

    # def get_data(self):
    #     training_dataset = datasets.CIFAR10(
    #         root="data",
    #         train=False,
    #         download=False,
    #         transform=ToTensor()
    #     )
    #     test_dataset = datasets.CIFAR10(
    #         root="data",
    #         train=False,
    #         download=False,
    #         transform=ToTensor()
    #     )
    #     data_portion = len(training_dataset) // self.num_nodes
    #     start_index = (self.port - 8000) * data_portion
    #     end_index = (self.port - 8000 + 1) * data_portion
    #     indexes = list(range(start_index, end_index))

    #     test_portion = len(test_dataset) // self.num_nodes
    #     test_start_index = (self.port - 8000) * test_portion
    #     test_end_index = (self.port - 8000 + 1) * test_portion
    #     test_indexes = list(range(test_start_index, test_end_index))

    #     self.training_dataset = torch.utils.data.Subset(training_dataset, indexes)
    #     self.test_dataset = torch.utils.data.Subset(test_dataset, test_indexes)

    #     self.training_dataloader = DataLoader(
    #         self.training_dataset, batch_size=self.batch_size)
    #     self.test_dataloader = DataLoader(
    #         self.test_dataset, batch_size=test_portion)

    def get_data(self):
        train_file = open(f"./data/femnist/train/node{self.port-8000}.json", "r")
        train_data = json.loads(train_file.read())
        test_file = open(f"./data/femnist/test/node{self.port-8000}.json", "r")
        test_data = json.loads(test_file.read())
        self.training_dataset, self.test_dataset = CustomImageDataset(train_data), CustomImageDataset(test_data)
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


    def load_model(self): 
        self.model.load_state_dict(torch.load("./models/global_client.pth"))

    def are_models_equal(self, model1, model2):
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()
        
        for key in model1_state_dict:
            if not torch.equal(model1_state_dict[key], model2_state_dict[key]):
                return False
        return True

    def train(self, server_port):
        if self.malicious:
            self.attack(server_port)
        else:
            self.model.train()
            losses = []
            for _ in range(self.epochs):
                epoch_loss = 0
                for batch, (X, y) in enumerate(self.training_dataloader):
                    X = X.to(self.device)
                    self.model = self.model.to(self.device)
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
                    grads = torch.tensor(json.loads(json.loads(res.content.decode())["grads"])).to(self.device)
                    epoch_loss += json.loads(res.content.decode())["loss"]
                    output.backward(grads)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(epoch_loss/len(self.training_dataloader))
            torch.save(self.model.state_dict(), f"./models/node_{self.port-8000}_client.pth")
            requests.post(f"http://localhost:{server_port}/server/round/",
                                            json={
                                                "client_port" : self.port,
                                                "losses" : losses
                                            })
    
    def attack(self, server_port):
        losses = []
        for _ in range(self.epochs):
            epoch_loss = 0
            for batch, (X, y) in enumerate(self.training_dataloader):
                X = torch.randn_like(X).to(self.device)
                self.model = self.model.to(self.device)
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
                epoch_loss += json.loads(res.content.decode())["loss"]
            losses.append(epoch_loss/len(self.training_dataloader))
        torch.save(self.model.state_dict(), f"./models/node_{self.port-8000}_client.pth")
        requests.post(f"http://localhost:{server_port}/server/round/",
                                        json={
                                            "client_port" : self.port,
                                            "losses" : losses
                                        })
        
    def predict(self, path):
        model = ClientNN().to("cpu")
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to("cpu")
                output = model(X)
                return output.clone().detach(), y