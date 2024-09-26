import os
import time
import subprocess
import requests

def run_node(i):
    # Run node{i}.py in the background and redirect output to a log file
    log_file = f"../logs/node_{i}.txt"
    with open(log_file, "w") as f:
        subprocess.Popen(
            ["python3", f"./node{i}.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

if __name__ == "__main__":
    num_nodes = 9
    cwd = os.path.dirname(__file__)

    # Step 1 : Bring up the network
    print("Bringing up the network...")
    # os.chdir(os.path.join(cwd, "test-network"))
    # os.system("./network.sh down")
    # os.system("sh ./start.sh")

    print("Bringing up the express applications...")
    os.chdir(os.path.join(cwd, "express-application"))
    with open("../logs/app1.txt", "w") as f:
        subprocess.Popen(
            ["node", f"./app1.js"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

    # Step 1: Bring up nodes (start processes)
    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        run_node(i)
        time.sleep(1)
    
    with open("../logs/aggregator.txt", "w") as f:
        subprocess.Popen(
            ["python3", f"./aggregator.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )
    time.sleep(1)

    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./req.sh")

    # Step 7: Re-initializing the model
    print("Re-initializing the global model...")
    os.chdir(os.path.join(cwd, "nodes"))
    os.system("python3 ./model.py")
    time.sleep(1)

    print("Starting the mining process...")
    requests.get("http://localhost:3000/start/")

    # The main script ends here, while the node processes continue running in the background
    print("Nodes have been started, and the script has completed.")