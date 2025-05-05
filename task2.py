import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Load MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Neural Network class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Train model with given seed
def train_model(seed):
    set_seed(seed)
    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_errors = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test error for this epoch
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for images, labels in test_loader:
                images = images.view(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_errors.append(test_loss / len(test_loader))

    return test_errors

# Run training for 5 different seeds
seeds = [0, 1, 2, 3, 4]
all_test_errors = []
final_test_errors = []

for i, seed in enumerate(seeds):
    print(f"\nTraining with seed {seed}")
    test_errors = train_model(seed)
    all_test_errors.append(test_errors)
    final_test_errors.append(test_errors[-1])

# Plot test error curves
for i, test_errors in enumerate(all_test_errors):
    plt.plot(test_errors, label=f"Seed {seeds[i]}")
plt.xlabel("Epoch")
plt.ylabel("Test Error")
plt.title("Test Error Across Random Seeds")
plt.legend()
plt.savefig("task2_test_error_seeds.png")
plt.show()

# Compute mean and std of final test error
final_test_errors = np.array(final_test_errors)
mean_err = final_test_errors.mean()
std_err = final_test_errors.std()
print(f"\nFinal Test Errors: {final_test_errors}")
print(f"Mean = {mean_err:.4f}, Std = {std_err:.4f}")
