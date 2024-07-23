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
    

class ServerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.classification_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 62),
            nn.Softmax(1)
        )

    def forward(self, data):
        x = self.conv_stack2(data)
        return self.classification_stack(x)
    

server = ServerNN()
client = ClientNN()
torch.save(server.state_dict(), "./models/global_server.pth")
torch.save(client.state_dict(), "./models/global_client.pth")