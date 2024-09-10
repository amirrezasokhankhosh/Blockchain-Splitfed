import matplotlib.pyplot as plt
import json


file_bl_split_fed = open("./blockchain_split_fed/losses/aggregator.json", 'r')
data_bl_split_fed = file_bl_split_fed.read()
bl_split_fed_cycles = json.loads(data_bl_split_fed)
bl_split_fed = []
for cycle in bl_split_fed_cycles:
    for _ in range(3):
        bl_split_fed.append(cycle)

file_split_fed = open("./split_fed/losses/scores.json", 'r')
data_split_fed = file_split_fed.read()
split_fed = json.loads(data_split_fed)

plt.plot(split_fed, label="SplitFed Learning")
plt.plot(bl_split_fed, label="Blockchain SplitFed")
plt.legend()
plt.show()