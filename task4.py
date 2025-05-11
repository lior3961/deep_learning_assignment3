import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import itertools

# Const Hyperparameters
input_size = 784
hidden_size = 128
num_classes = 10

# Model
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

def train_with_dynamic_hyperparameter(learning_rate, batch_size, num_epochs, train_dataset, val_dataset, test_dataset):
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNet(input_size, hidden_size, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_errors = []
    validation_errors = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation error
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.view(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            validation_errors.append(val_loss / len(val_loader))

            # Test error
            test_loss = 0
            for images, labels in test_loader:
                images = images.view(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_errors.append(test_loss / len(test_loader))

            

    min_val_index = np.argmin(validation_errors)
    e_va_star = validation_errors[min_val_index]
    e_te_star = test_errors[min_val_index]

    return e_va_star, e_te_star

# Load and split MNIST dataset
transform = transforms.ToTensor()
train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Create train/val split: 50,000 for training, 10,000 for validation
train_dataset, val_dataset = random_split(train_dataset_full, [50000, 10000])

learning_rates = [0.001, 0.01, 0.0001]
batch_sizes = [100, 200]
num_epochs = [3,5,7]

te_star_list = []
va_star_list = []

for lr, bs, ne in itertools.product(learning_rates, batch_sizes, num_epochs):
    learning_rates.append(lr)
    batch_sizes.append(bs)
    num_epochs.append(ne)
    e_va_star, e_te_star = train_with_dynamic_hyperparameter(lr, bs, ne, train_dataset, val_dataset, test_dataset)
    va_star_list.append(e_va_star)
    te_star_list.append(e_te_star)
    print("Validation errors:", va_star_list)
    print("Test errors:", te_star_list)
    print(f"Training with lr={lr}, bs={bs}, ne={ne}: e_va*={e_va_star}")

print("Validation errors:", va_star_list)
min_val_index = np.argmin(va_star_list)
best_learning_rate = learning_rates[min_val_index]
best_batch_size = batch_sizes[min_val_index]
best_num_epochs = num_epochs[min_val_index]
best_e_te_star = te_star_list[min_val_index]
print(f"Best hyperparameters: learning_rate={best_learning_rate}, batch_size={best_batch_size}, num_epochs={best_num_epochs}")
print(f"Best test error: e_te*={best_e_te_star}")

