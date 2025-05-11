
import torch
import torch.nn 
import torch.nn.functional 
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot 

# Hyperparameters
input_size = 784
hidden_size = 128
num_classes = 10

# Define model
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, output_size=num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

    def extract_hidden(self, x):
        x = x.view(x.size(0), -1)
        return torch.torch.nn.functional.relu(self.fc1(x))

    def flatten_input(self, x):
        return x.view(x.size(0), -1)

# Load combined MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
full_dataset = ConcatDataset([train_dataset, test_dataset])
full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMLP().to(device)
model.eval()

# Extract hidden features and raw flattened inputs
hidden_features = []
raw_inputs = []
all_labels = []

with torch.no_grad():
    for images, labels in full_loader:
        images = images.to(device)
        hidden = model.extract_hidden(images)
        flat = model.flatten_input(images)

        hidden_features.append(hidden.cpu())
        raw_inputs.append(flat.cpu())
        all_labels.append(labels)

hidden_tensor = torch.cat(hidden_features, dim=0)
raw_tensor = torch.cat(raw_inputs, dim=0)
labels_tensor = torch.cat(all_labels, dim=0)

# t-SNE for hidden features
print("Rutorch.nning t-SNE on hidden features...")
tsne_hidden = TSNE(n_components=2, random_state=42)
hidden_2d = tsne_hidden.fit_transform(hidden_tensor.numpy())

matplotlib.pyplot.figure(figsize=(10, 8))
scatter = matplotlib.pyplot.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=labels_tensor.numpy(), cmap='tab10', s=5)
matplotlib.pyplot.colorbar(scatter, ticks=range(10))
matplotlib.pyplot.title("t-SNE of First-Layer Hidden Features")
matplotlib.pyplot.savefig("tsne_hidden_features.png")

# t-SNE for raw input
print("Rutorch.nning t-SNE on raw inputs...")
tsne_raw = TSNE(n_components=2, random_state=42)
raw_2d = tsne_raw.fit_transform(raw_tensor.numpy())

matplotlib.pyplot.figure(figsize=(10, 8))
scatter = matplotlib.pyplot.scatter(raw_2d[:, 0], raw_2d[:, 1], c=labels_tensor.numpy(), cmap='tab10', s=5)
matplotlib.pyplot.colorbar(scatter, ticks=range(10))
matplotlib.pyplot.title("t-SNE of Raw Input Pixels")
matplotlib.pyplot.savefig("tsne_raw_inputs.png")
