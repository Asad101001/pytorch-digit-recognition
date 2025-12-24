import torch  # Core PyTorch library for tensors and NN
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimizers (e.g., Adam)
import torchvision  # For datasets and image transforms
import torchvision.transforms as transforms  # For preprocessing images
import matplotlib.pyplot as plt  # For plotting results
import numpy as np  # For numerical operations

# Step 1: Data Preparation (Tensor Conversion and Loading)
# MNIST images are 28x28 grayscale. We convert them to tensors and normalize (scale to 0-1).
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to tensor (shape: [1, 28, 28] for grayscale)
    transforms.Normalize((0.5,), (0.5,))  # Normalizes: (pixel - 0.5) / 0.5, so values are -1 to 1
])

# Load MNIST dataset (downloads if not present)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders: Split data into batches for efficient training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  # 64 images per batch, shuffled
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Explanation: Tensors are PyTorch's data structure. We converted images (from PIL format) to tensors for NN processing.

# Step 2: Define the Neural Network Model (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer: Detects features (e.g., edges) in images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input: 1 channel (grayscale), Output: 32 features
        # Fully connected layer: Classifies features into 10 digits
        self.fc1 = nn.Linear(32 * 28 * 28, 10)  # 32 features * 28x28 image size -> 10 outputs (0-9)
        self.relu = nn.ReLU()  # Activation: Makes output non-negative for better learning

    def forward(self, x):
        # Forward pass: How data flows through the network
        x = self.relu(self.conv1(x))  # Convolution + ReLU
        x = x.view(x.size(0), -1)  # Flatten: Turn 2D features into 1D for fully connected layer
        x = self.fc1(x)  # Output logits (raw scores for each digit)
        return x

model = SimpleCNN()

# Explanation: This is a basic CNN. Conv layer scans the image with filters. ReLU adds non-linearity. FC layer predicts digits.

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Loss: Measures error for classification (lower is better)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer: Updates model weights; lr=0.001 is learning rate

# Explanation: Loss tells us how wrong predictions are. Optimizer adjusts weights to minimize loss.

# Step 4: Training the Model
num_epochs = 5  # Train for 5 full passes over data
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:  # Loop over batches
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass: Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass: Compute gradients
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Explanation: Training loop: Forward (predict), compute loss, backward (learn), update. Repeat for epochs.

# Step 5: Evaluate the Model
model.eval()  # Set to evaluation mode (no training)
correct = 0
total = 0
with torch.no_grad():  # No gradients needed for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted digit (highest score)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

# Step 6: Visualize a Prediction
# Get one test image
dataiter = iter(test_loader)
images, labels = next(dataiter)
img = images[0].squeeze()  # Remove batch dimension
plt.imshow(img, cmap='gray')  # Show image
plt.title(f'Predicted: {torch.argmax(model(images[0].unsqueeze(0))).item()}, Actual: {labels[0].item()}')
plt.show()

# Step 7: Save the Model
torch.save(model.state_dict(), 'mnist_cnn.pth')  # Save trained weights
print("Model saved as mnist_cnn.pth")