import os

# List of different malicious client settings to test
malicious_settings = [0, 2, 5, 10, 15]

# Number of total clients and rounds
num_clients = 20
rounds = 10

# Run experiments
for num_malicious in malicious_settings:
    print(f"Running experiment: {num_malicious} malicious clients out of {num_clients}")
    os.system(f"python main.py --num_clients {num_clients} --num_malicious {num_malicious} --rounds {rounds}")
