import matplotlib.pyplot as plt
import json
from glob import glob

num_servers = 3
num_rows = 1

filenames = glob("./losses/*")
fig, axs = plt.subplots(num_rows, num_servers)
client_colors = ['C0', 'C1']

for i in range(len(filenames)):
    file = open(filenames[i])
    data = json.loads(file.read())
    for j, (client, rounds) in enumerate(data.items()):
        batch_range = 0
        for each_round, losses in rounds.items():
            axs[i].plot(range(batch_range, batch_range + len(losses)), losses, label=f"Client {int(client) - 8000}", color=client_colors[j])
            axs[i].set_title(f"Server {i}")
            axs[i].legend()
            batch_range += len(losses)

axs[1].set_xlabel("Batches")
plt.show()