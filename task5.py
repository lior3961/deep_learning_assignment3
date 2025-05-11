import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split, ConcatDataset

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
        return out


# Load and split MNIST dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)


# Combine them
full_dataset = ConcatDataset([train_dataset, test_dataset])

# use tNSE to visualize the dataset
def visualize_dataset(dataset):
    data_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
    images, labels = next(iter(data_loader))
    images = images.view(-1, 28*28).numpy()
    labels = labels.numpy()
    #color every label with a different color
    
    tsne = TSNE(n_components=2, random_state=0)
    images_2d = tsne.fit_transform(images)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(images_2d[:, 0], images_2d[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of MNIST dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()