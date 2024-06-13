import matplotlib.pyplot as plt
import json


file = open("./losses.json")

data = json.loads(file.read())

for client, rounds in data.items():
    batch_range = 0
    for each_round, losses in rounds.items():
        
        plt.plot(range(batch_range, batch_range + len(losses)), losses, label=f"Client {int(client) - 8000}")
        batch_range += len(losses)

plt.legend()
plt.xlabel("batches")
plt.ylabel("Cross entropy loss")
plt.show()