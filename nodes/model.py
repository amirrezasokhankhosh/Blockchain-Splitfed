from global_var import *    


server = ServerNN()
client = ClientNN()
torch.save(server.state_dict(), "./models/global_server.pth")
torch.save(client.state_dict(), "./models/global_client.pth")