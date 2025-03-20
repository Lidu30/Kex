import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from kan1 import KAN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Define dataset class
class SchizophreniaEEGDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load EEG data (each file has 16 channels, 7680 samples per channel)
        data = np.loadtxt(file_path).reshape(16, 7680)  # Shape: (16, 7680)
        data = data.T  # Shape: (7680, 16) for compatibility with KAN
        
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)

# Load data paths
schizo_path = "healthy"
healthy_path = "sch"

schizo_files = [os.path.join(schizo_path, f) for f in os.listdir(schizo_path) if f.endswith(".eea")]
healthy_files = [os.path.join(healthy_path, f) for f in os.listdir(healthy_path) if f.endswith(".eea")]

file_paths = schizo_files + healthy_files
labels = [1] * len(schizo_files) + [0] * len(healthy_files)  # 1 = Schizophrenia, 0 = Healthy

# Train-test split
train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Create datasets and loaders
batch_size = 32
train_dataset = SchizophreniaEEGDataset(train_paths, train_labels)
test_dataset = SchizophreniaEEGDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model (KAN input size adjusted to 7680 * 16)
model = KAN([7680 * 16, 64, 32, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer, loss function, scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 20
best_auroc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in train_loader:
        data = data.view(data.size(0), -1).to(device)  # Flatten input
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    scheduler.step()
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_preds)
    print(f"Validation AUROC: {auroc:.4f}")
    
    if auroc > best_auroc:
        best_auroc = auroc
        torch.save(model.state_dict(), "best_kan_schizo.pth")
        print("Best model saved!")

print("Training Complete!")
