import torch
import copy
import torch.nn.functional as F

def fedrola_aggregate(global_model, client_models, threshold):
    # Only use weights and biases for flattening
    global_flat = torch.cat([v.flatten() for k, v in global_model.state_dict().items() if "weight" in k or "bias" in k])
    valid_updates = []

    for client_model in client_models:
        client_flat = torch.cat([v.flatten() for k, v in client_model.state_dict().items() if "weight" in k or "bias" in k])
        similarity = F.cosine_similarity(global_flat, client_flat, dim=0)
        if similarity > threshold:
            valid_updates.append(client_model.state_dict())

    if not valid_updates:
        print("Warning: No valid updates! Using global model.")
        return global_model.state_dict()

    avg_update = copy.deepcopy(valid_updates[0])
    for k in avg_update.keys():
        for i in range(1, len(valid_updates)):
            avg_update[k] += valid_updates[i][k]
        avg_update[k] = avg_update[k].float()
        avg_update[k] /= len(valid_updates)

    return avg_update
