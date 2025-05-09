import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def plot_results(accuracies, args):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    for i, accuracy in enumerate(accuracies):
        plt.text(i + 1, accuracy, f'{accuracy:.2f}%', ha='center', va='bottom', fontsize=9)
    plt.title(f"FedRoLa Accuracy | {args.num_malicious}/{args.num_clients} malicious clients")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    save_path = (
        f"plots/fedrola_accuracy_{args.model}_"
        f"{args.num_malicious}_malicious_"
        f"{args.num_clients}_clients_"
        f"{args.local_epochs}_epochs_"
        f"{args.rounds}_rounds.png"
    )
    plt.savefig(save_path)
    plt.show()
    
    # --- Save CSV file ---
    csv_filename = (
        f"plots/fedrola_accuracy_{args.model}_"
        f"{args.num_malicious}_malicious_"
        f"{args.num_clients}_clients_"
        f"{args.local_epochs}_epochs_"
        f"{args.rounds}_rounds.csv"
    )

    df = pd.DataFrame({
        'Round': list(range(1, len(accuracies) + 1)),
        'TestAccuracy': accuracies
    })
    df.to_csv(csv_filename, index=False)