import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

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


# Load and split MNIST dataset
transform = transforms.ToTensor()
train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Create train/val split: 50,000 for training, 10,000 for validation
train_dataset, val_dataset = random_split(train_dataset_full, [50000, 10000])

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Train the model and return e_te*, e_va*, and error curves
def train_model_with_dynamic_hyperparameters(seed, learning_rate=0.001, num_epochs=5, train_loader=None, val_loader=None, test_loader=None):
    set_seed(seed)
    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_errors = []
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

        # Validation error
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.view(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_errors.append(val_loss / len(val_loader))

            # Test error
            test_loss = 0
            for images, labels in test_loader:
                images = images.view(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_errors.append(test_loss / len(test_loader))

    min_val_index = np.argmin(val_errors)
    e_va_star = val_errors[min_val_index]
    e_te_star = test_errors[min_val_index]
    return e_te_star, e_va_star, val_errors, test_errors

def train_by_hyperparameter(seed, learning_rate, batch_size, num_epochs, te_star_list, va_star_list):
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    e_te_star, e_va_star, val_errors, test_errors = train_model_with_dynamic_hyperparameters(seed, learning_rate, batch_size, num_epochs, train_loader, val_loader, test_loader)
    te_star_list.append(e_te_star)
    va_star_list.append(e_va_star)

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [100, 150, 200]
num_epochs = [3,5,7]

te_star_list = []
va_star_list = []

for lr in learning_rates:
    for bs in batch_sizes:
        for ne in num_epochs:
            print(f"Training with lr={lr}, bs={bs}, ne={ne}")
            train_by_hyperparameter(0, lr, bs, ne, te_star_list, va_star_list)

te_star_array = np.array(te_star_list)
mean_te = te_star_array.mean()
std_te = te_star_array.std()