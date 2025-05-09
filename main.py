import torch
import argparse
import os
import copy
from models import SmallCNN, ResNet18_CIFAR
from client import client_update
from server import fedrola_aggregate
from utils import load_cifar10, evaluate, plot_results
from tqdm import tqdm

def main(args):
    device = torch.device('cpu')  # You can later change to GPU if available
    os.makedirs("plots", exist_ok=True)

    # Load datasets
    trainset, testset = load_cifar10()
    client_datasets = torch.utils.data.random_split(trainset, [len(trainset)//args.num_clients]*args.num_clients)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Initialize model
    if args.model == 'smallcnn':
        global_model = SmallCNN().to(device)
    elif args.model == 'resnet18':
        global_model = ResNet18_CIFAR().to(device)
    else:
        raise ValueError("Model must be 'smallcnn' or 'resnet18'.")

    test_accuracies = []

    # Federated learning rounds
    for rnd in range(args.rounds):
        print(f"Round {rnd+1}/{args.rounds}")
        client_models = []
        malicious_ids = set(range(args.num_malicious))

        for cid in tqdm(range(args.num_clients), desc="Clients"):
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
            loader = torch.utils.data.DataLoader(client_datasets[cid], batch_size=32, shuffle=True)
            is_malicious = cid in malicious_ids
            client_update(local_model, optimizer, loader, device, malicious=is_malicious, local_epochs=args.local_epochs)
            client_models.append(local_model)

        # FedRoLa aggregation
        new_state = fedrola_aggregate(global_model, client_models, threshold=0.3)
        global_model.load_state_dict(new_state)

        # Evaluate
        acc = evaluate(global_model, test_loader, device) * 100
        print(f"Test Accuracy: {acc:.2f}%")
        test_accuracies.append(acc)

    # Plot final results
    plot_results(test_accuracies, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_malicious", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
