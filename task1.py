import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Ensures that random initializations are the same every run.
torch.manual_seed(0)

# Hyperparameters
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# dataset download, each image becomes a tensor of shape [1, 28, 28]
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

#Batches the dataset and optionally shuffles the training data.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define model
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

model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_errors = []
test_errors = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.view(-1, 28*28)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_errors.append(total_loss / len(train_loader))

    # Evaluate test error
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        test_errors.append(test_loss / len(test_loader))
        print(f"Epoch {epoch+1}, Train Error: {train_errors[-1]:.4f}, Test Error: {test_errors[-1]:.4f}")

# Plot error curves
plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs. Test Error")
plt.legend()
plt.show()

# Show some misclassified images
model.eval()
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images_flat = images.view(-1, 28*28)
        outputs = model(images_flat)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified.append((images[i], predicted[i], labels[i]))
            if len(misclassified) >= 10:
                break
        if len(misclassified) >= 10:
            break

# Display misclassified examples
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i, (img, pred, label) in enumerate(misclassified):
    ax = axs[i // 5, i % 5]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Pred: {pred.item()} / True: {label.item()}")
    ax.axis('off')
plt.suptitle("Misclassified Images")
plt.show()
