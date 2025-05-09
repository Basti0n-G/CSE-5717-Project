import pandas as pd
import matplotlib.pyplot as plt

# Load results
smallcnn = pd.read_csv('plots/fedrola_accuracy_smallcnn_6_malicious_8_clients_3_epochs_10_rounds.csv')
resnet18 = pd.read_csv('plots/fedrola_accuracy_resnet18_6_malicious_8_clients_3_epochs_10_rounds.csv')

# Plot together
plt.figure()
plt.plot(smallcnn['Round'], smallcnn['TestAccuracy'], label='SmallCNN', marker='o')
plt.plot(resnet18['Round'], resnet18['TestAccuracy'], label='Resnet18', marker='s')
plt.xlabel('Round')
plt.ylabel('Test Accuracy (%)')
plt.title('FedRoLa Comparison - SmallCNN vs Resnet18 (6 malicious clients)')
plt.grid(True)
plt.legend()
plt.savefig('plots/comparison_smallcnn_vs_resnet18_6_malicious_8_clients.png')
plt.show()
