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
    

server = ServerNN()
client = ClientNN()
torch.save(server.state_dict(), "./models/global_server.pth")
torch.save(client.state_dict(), "./models/global_client.pth")