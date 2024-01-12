import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
dataset = datasets.ImageFolder('/Users/protim/Downloads/dataset/', transform=transform)

# Define k-fold cross-validation
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize variables for performance metrics across folds and epochs
avg_metrics = {
    'accuracy': [],
    'macro_precision': [],
    'macro_recall': [],
    'macro_f1': [],
    'micro_precision': [],
    'micro_recall': [],
    'micro_f1': [],
}

for fold, (train_index, test_index) in enumerate(skf.split(dataset.samples, dataset.targets), 1):
    # Create data loaders for the current fold
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)

    # Define and train the model (using mainCNNModel as an example)
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


    model = mainCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize variables for metrics within the current fold
    fold_metrics = {
        'accuracy': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
        'micro_precision': [],
        'micro_recall': [],
        'micro_f1': [],
    }

    # Training loop (similar to the previous code)
    epochs = 1  # You can adjust the number of epochs
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

        # Evaluate the model on the test set after each epoch
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        # Calculate performance metrics for the current epoch
        accuracy = accuracy_score(all_labels, all_preds)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

        # Store metrics for the current epoch
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['macro_precision'].append(macro_precision)
        fold_metrics['macro_recall'].append(macro_recall)
        fold_metrics['macro_f1'].append(macro_f1)
        fold_metrics['micro_precision'].append(micro_precision)
        fold_metrics['micro_recall'].append(micro_recall)
        fold_metrics['micro_f1'].append(micro_f1)

        # Confusion Matrix for the current epoch
        '''cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
        plt.title(f"Confusion Matrix - Fold {fold}, Epoch {epoch + 1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()'''

    # Calculate and print average metrics for the current fold
    avg_fold_metrics = {metric: sum(values) / len(values) for metric, values in fold_metrics.items()}
    print(f"Average Metrics - Fold {fold}:\n{avg_fold_metrics}\n")

    # Store metrics for the current fold
    for metric in avg_metrics.keys():
        avg_metrics[metric].append(avg_fold_metrics[metric])

# Calculate and print average metrics across all folds
avg_metrics_final = {metric: sum(values) / len(values) for metric, values in avg_metrics.items()}
print(f"Average Metrics Across All Folds:\n{avg_metrics_final}")