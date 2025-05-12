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
        out = self.fc2(out)
        return out
    
# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# use tNSE to visualize the dataset
def visualize_dataset(dataset):
    data_loader = DataLoader(dataset, batch_size=6000, shuffle=True)
    images, labels = next(iter(data_loader))
    images = images.view(-1, 28*28).numpy()
    labels = labels.numpy()
    tsne = TSNE(n_components=2, random_state=0)
    images_2d = tsne.fit_transform(images)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(images_2d[:, 0], images_2d[:, 1], c=labels, cmap=plt.colormaps['tab10'], alpha=0.25)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE visualization of MNIST dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# use tNSE to visualize the hidden layer of the dataset
def visualize_hidden_layer(model, dataset):
    data_loader = DataLoader(dataset, batch_size=6000, shuffle=True)
    images, labels = next(iter(data_loader))
    images = images.view(-1, 28*28)

    with torch.no_grad():
        hidden_layer_output = model.relu(model.fc1(images)).numpy()

    tsne = TSNE(n_components=2, random_state=0)
    hidden_layer_2d = tsne.fit_transform(hidden_layer_output)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(hidden_layer_2d[:, 0], hidden_layer_2d[:, 1], c=labels.numpy(), cmap=plt.colormaps['tab10'], alpha=0.25)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE visualization of hidden layer output')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


# Load and split MNIST dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
set_seed(1)
# visualize_dataset(train_dataset)

model = NeuralNet(input_size, hidden_size, num_classes)
set_seed(1)
visualize_hidden_layer(model, train_dataset)
