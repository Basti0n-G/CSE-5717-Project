# FedRoLa-Experiment

Simulates Federated Learning under adversarial attacks using the FedRoLa defense algorithm.

## Project Structure

- `models.py`: VGG-11 model architecture.
- `client.py`: Local client training and malicious behavior.
- `server.py`: Server aggregation with FedRoLa defense.
- `attacks.py`: Placeholder for future custom adversarial attacks.
- `utils.py`: Utilities (dataset loading, evaluation, plotting).
- `main.py`: Main script to run federated training.
- `run_experiments.py`: Automatically run experiments with different numbers of malicious clients.
- `requirements.txt`: Required Python packages.
- `plots/`: Auto-saved graphs from experiments.
- `data/cifar-10-python.tar.gz`: CIFAR-10 dataset (Too big to be included in repo).

## Installation

```bash
pip install -r requirements.txt
