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

    # Step 1: Bring up nodes (start processes)
    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        run_node(i)
        time.sleep(1)
    
    with open("../logs/fed_server.txt", "w") as f:
        subprocess.Popen(
            ["python3", f"./fed_server.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

    # Step 7: Re-initializing the model
    print("Re-initializing the global model...")
    os.system("python3 ./model.py")
    time.sleep(1)

    print("Starting the mining process...")
    requests.get("http://localhost:5050/start/")

    # The main script ends here, while the node processes continue running in the background
    print("Nodes have been started, and the script has completed.")