import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.image import imread
import pathlib
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
train_data = datasets.ImageFolder('/Users/protim/Downloads/dataset_part3/train/senior/', transform=transform)
val_data = datasets.ImageFolder('/Users/protim/Downloads/dataset/val', transform=transform)
test_data = datasets.ImageFolder('/Users/protim/Downloads/dataset/test', transform=transform)

# Split data
train_size = int(0.7 * len(train_data))
val_size = int(0.15 * len(train_data))
test_size = len(train_data) - train_size - val_size

train_set, val_set, _ = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

image_folder = ['angry', 'bored', 'engaged', 'neutral']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('/Users/protim/Downloads/dataset_part3/train/senior/' + i + '/'))
    nimgs[i] = nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Training Dataset')
plt.show()

'''image_folder = ['angry', 'bored', 'engaged', 'neutral']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('/Users/protim/Downloads/dataset_part3/train/senior/' + i + '/'))
    nimgs[i] = nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Training Dataset')
plt.show()'''

for i in ['angry', 'bored', 'engaged', 'neutral']:
    print('Training {} images are: '.format(i) + str(
        len(os.listdir('/Users/protim/Downloads/dataset_part3/train/senior/' + i + '/'))))

class mainCNNModel(nn.Module):
    def __init__(self):
        super(mainCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # for 4 classes, added an extra fully connected layer

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

class variant_1_CNNModel(nn.Module):
    def __init__(self):
        super(variant_1_CNNModel, self).__init__()
        # Modify hyperparameters or architecture for variant_1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Change kernel size or stride or padding
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)  # Change dropout rate
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 256 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class variant_2_CNNModel(nn.Module):
    def __init__(self):
        super(variant_2_CNNModel, self).__init__()
        # Modify hyperparameters or architecture for variant_2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # Change kernel size or stride or padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)  # Change dropout rate
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Choose the model to use (mainCNNModel, variant_1_CNNModel, or variant_2_CNNModel)
# Choose the model to use (mainCNNModel, variant_1_CNNModel, or variant_2_CNNModel)
model = mainCNNModel()

# Check if a saved model exists
# if os.path.exists('best_face_expression_model.pth'):
    # Load the best model
    # model.load_state_dict(torch.load('best_face_expression_model.pth'))
    # print("Loaded the best model.")
# model = variant_1_CNNModel()
# model = variant_2_CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,             # Learning rate: Controls the step size during optimization.
    betas=(0.9, 0.999),    # Betas are coefficients used for computing running averages of gradient and its square.
    eps=1e-8,              # Epsilon: A small constant for numerical stability in the denominator of the update rule.
    weight_decay=1e-4,     # Weight decay: L2 regularization term to prevent overfitting.
    amsgrad=False          # AMSGrad variant. If True, uses the corrected version of the Adam update rule.
)

# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement before stopping
counter = 0  # Counter to track the number of epochs without improvement

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

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_face_expression_model.pth')
        else:
            counter += 1

        # Print and display metrics
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}")
        print("Validation Metrics:")
        print(f"Best Validation Loss: {best_val_loss}, Patience Counter: {counter}")
        print("")

        # Stop training if patience is reached
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to lack of improvement.")
            break

# Load the best-performing model
best_model = mainCNNModel()
best_model.load_state_dict(torch.load('best_face_expression_model.pth'))
best_model.eval()

# Macro and Micro Precision, Recall, F1 Score
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}")
print(f"Macro Precision: {precision_macro}, Macro Recall: {recall_macro}, Macro F1: {f1_macro}")
print(f"Micro Precision: {precision_micro}, Micro Recall: {recall_micro}, Micro F1: {f1_micro}")
print("Confusion Matrix:")
print(cm)


'''# Define the function to test a single image
def test_single_image(image_path, model, class_to_idx=None):
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
        _, predicted_idx = torch.max(output, 1)

        # Map indices to class names
        if class_to_idx is not None:
            idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
            predicted_class = idx_to_class[predicted_idx.item()]
            # print(f"Predicted class: {predicted_class}")
        else:
            print(f"Predicted class index: {predicted_idx.item()}")

        return predicted_class  # Return the predicted class index

# Replace 'path_to_your_image.jpg' with the actual path of your test image
image_path = '/Users/protim/Downloads/dataset/test/angry/angry6.jpg'

# Load your model (assuming you've already defined and loaded your model)
#loaded_model = mainCNNModel()  # Replace CNNModel() with your actual model loading code

loaded_model = mainCNNModel()
loaded_model.load_state_dict(torch.load('best_face_expression_model.pth'))
loaded_model.eval()

# Test the image and get the predicted class
predicted_class = test_single_image(image_path, loaded_model, class_to_idx=test_data.class_to_idx)

print(f"Predicted class: {predicted_class}")
'''




