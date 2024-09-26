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

    # Step 1: Bring up miners (start processes)
    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        run_node(i)
        time.sleep(1)

    # Step 7: Re-initializing the model
    print("Re-initializing the global model...")
    os.system("python3 ./model.py")
    time.sleep(1)

    # Step 8: Sending start request
    clients = []
    for i in range(num_nodes - 1):
        client = {"port": 8000 + i + 1}
        clients.append(client)

    print("Starting the mining process...")
    requests.post("http://localhost:8000/server/", json={
        "clients": clients,
        "cycle": 0
    })

    # The main script ends here, while the node processes continue running in the background
    print("Nodes have been started, and the script has completed.")
