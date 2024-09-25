import os
import time
import requests
import concurrent.futures


if __name__ == "__main__":
    num_nodes = 9
    cwd = os.path.dirname(__file__)
    executor = concurrent.futures.ProcessPoolExecutor(14)

    # Step 1 : Bring up the network
    # print("Bringing up the network...")
    # os.chdir(os.path.join(cwd, "test-network"))
    # os.system("sh ./start.sh")

    # Step 2 : Bring up express applications
    print("Bringing up the express applications...")
    os.chdir(os.path.join(cwd, "express-application"))
    executor.submit(os.system, "node ./app1.js > ../logs/app1.txt")
    time.sleep(3)

    # Step 3 : Bring up miners
    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        executor.submit(os.system, f"python3 ./node{i}.py > ../logs/node_{i}.txt")
        time.sleep(3)

    # Step 4 : Bring up Aggregator
    print("Bringing up the aggregator...")
    executor.submit(os.system, "python3 ./aggregator.py > ../logs/aggregator.txt")
    time.sleep(3)

    # Step 4 : Initialize ledgers
    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./req.sh")

    # Step 7 : Re-initializing the model
    print("Re-initializing the global model...")
    os.chdir(os.path.join(cwd, "nodes"))
    os.system("python3 ./model.py")
    time.sleep(3)

    # Step 8 : Sending start request
    print("Starting the mining process...")
    requests.post("http://localhost:3000/start/")