import matplotlib.pyplot as plt
import json

file = open("./winners.json", 'r')
data = file.read();

cycles = json.loads(data)

winners = {}
for i in range(1, 21):
    servers = cycles[f"cycle_{i}"]["servers"]
    for server in servers:
        if server not in winners.keys():
            winners[server] = 1
        else:
            winners[server] += 1
            
winners = dict(sorted(winners.items()))
plt.bar(winners.keys(), winners.values())
plt.show()