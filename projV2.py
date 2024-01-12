import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
train_data = datasets.ImageFolder('/Users/protim/Downloads/dataset/train', transform=transform)
val_data = datasets.ImageFolder('/Users/protim/Downloads/dataset/val', transform=transform)
test_data = datasets.ImageFolder('/Users/protim/Downloads/dataset/test', transform=transform)

# Split data
train_size = int(0.7 * len(train_data))
val_size = int(0.15 * len(train_data))
test_size = len(train_data) - train_size - val_size

train_set, val_set, _ = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

import torch.nn as nn


import torch.nn as nn

class CNNModel(nn.Module):
    def _init_(self):
        super(CNNModel, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Added dropout layer
        self.batch_norm1 = nn.BatchNorm2d(32)  # Added batch normalization
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)  # Changed the second fully connected layer
        self.fc3 = nn.Linear(256, 4)  # Assuming 4 classes, added an extra fully connected layer

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Added dropout
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}")
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        print("Confusion Matrix:")
        print(cm)

# Save model
torch.save(model.state_dict(), 'face_expression_model.pth')

# Load model
loaded_model = CNNModel()
loaded_model.load_state_dict(torch.load('face_expression_model.pth'))
loaded_model.eval()

# Example of testing with a single image

from PIL import Image
import torch
from torchvision import transforms


# Define the function to test a single image
def test_single_image(image_path, model):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Return the predicted class


# Replace 'path_to_your_image.jpg' with the actual path of your test image
image_path = '/Users/protim/Downloads/dataset/test/bored/bored1.jpg'

# Load your model (assuming you've already defined and loaded your model)
loaded_model = CNNModel()  # Replace CNNModel() with your actual model loading code

# Test the image and get the predicted class
predicted_class = test_single_image(image_path, loaded_model)

print(f"Predicted class: {predicted_class}")
