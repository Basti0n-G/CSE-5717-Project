import torch
import torch.nn.functional as F

def client_update(model, optimizer, train_loader, device, malicious=False, local_epochs=1):
    model.train()
    for _ in range(local_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            if malicious:
                for param in model.parameters():
                    param.grad.data *= -5
            optimizer.step()
    return model
