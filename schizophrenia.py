import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import argparse
import numpy as np
from kan1 import KAN
from datasets import create_schizophrenia_datasets

# Parse command line arguments
parser = argparse.ArgumentParser(description='Schizophrenia EEG Classification')
parser.add_argument('--healthy_dir', type=str, required=True,
                    help='Path to the directory containing healthy subject data')
parser.add_argument('--schizophrenia_dir', type=str, required=True,
                    help='Path to the directory containing schizophrenia subject data')
parser.add_argument('--output_dir', type=str, default='./Results/Schizophrenia/',
                    help='Path to save results')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='Size of the hidden layers')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--load', type=str, default='',
                    help='Path to load a pre-trained model')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Create metrics directories
metrics_dir = os.path.join(output_dir, 'metrics')
auroc_dir = os.path.join(output_dir, 'AUROC')
auprc_dir = os.path.join(output_dir, 'AUPRC')
cm_dir = os.path.join(output_dir, 'CM')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(auroc_dir, exist_ok=True)
os.makedirs(auprc_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)

# Load and prepare datasets
print("Loading and preparing datasets...")
train_loader, test_loader, n_classes = create_schizophrenia_datasets(
    args.healthy_dir, 
    args.schizophrenia_dir,
    test_size=0.2,
    batch_size=args.batch_size
)

# Get input size from the first batch
for images, _ in train_loader:
    input_size = images.size(1)
    break

print(f"Input size: {input_size}")

# Define model
# The first layer input size should match your data's flattened dimensions (16 channels * 23 freqs * 125 time steps)
model = KAN([input_size, args.hidden_size, args.hidden_size, n_classes])

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model if specified
if args.load:
    print(f"Loading pre-trained model from {args.load}")
    model = torch.load(args.load)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

# Move model to device
model.to(device)

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score, epoch):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc="lower right")
    
    # Save the plot
    plt.savefig(os.path.join(auroc_dir, f'roc_epoch_{epoch}.png'))
    plt.close()

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_score, epoch):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    pr_auc = metrics.average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Epoch {epoch}')
    plt.legend(loc="lower left")
    
    # Save the plot
    plt.savefig(os.path.join(auprc_dir, f'pr_epoch_{epoch}.png'))
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.colorbar()
    
    classes = ['Healthy', 'Schizophrenia']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot
    plt.savefig(os.path.join(cm_dir, f'cm_epoch_{epoch}.png'))
    plt.close()

# Function to save training metrics
def save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_auroc, val_precision, val_recall, val_f1):
    metrics_file = os.path.join(metrics_dir, 'training_metrics.txt')
    
    with open(metrics_file, 'a') as f:
        metrics_str = (f'Epoch: {epoch:3d}, '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.4f}, '
                       f'Val AUROC: {val_auroc:.4f}, '
                       f'Val Precision: {val_precision:.4f}, '
                       f'Val Recall: {val_recall:.4f}, '
                       f'Val F1: {val_f1:.4f}\n')
        f.write(metrics_str)

# Function to plot training progress
def plot_training_progress():
    metrics_file = os.path.join(metrics_dir, 'training_metrics.txt')
    
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_aurocs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.startswith('Epoch'):
                parts = line.strip().split(',')
                epochs.append(int(parts[0].split(':')[1].strip()))
                train_losses.append(float(parts[1].split(':')[1].strip()))
                train_accs.append(float(parts[2].split(':')[1].strip()))
                val_losses.append(float(parts[3].split(':')[1].strip()))
                val_accs.append(float(parts[4].split(':')[1].strip()))
                val_aurocs.append(float(parts[5].split(':')[1].strip()))
                val_precisions.append(float(parts[6].split(':')[1].strip()))
                val_recalls.append(float(parts[7].split(':')[1].strip()))
                val_f1s.append(float(parts[8].split(':')[1].strip()))
    
    # Plot accuracy
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # Plot AUROC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_aurocs)
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('AUROC vs. Epoch')
    
    # Plot F1, Precision, Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_precisions, label='Precision')
    plt.plot(epochs, val_recalls, label='Recall')
    plt.plot(epochs, val_f1s, label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'training_progress.png'))
    plt.close()

# Variables to track best model
best_auroc = 0
best_f1 = 0

# Training loop
print("Starting training...")
for epoch in range(args.epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * train_correct / train_total,
                'lr': optimizer.param_groups[0]['lr']
            })
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # Lists to store predictions and true labels
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(output.softmax(dim=1)[:, 1].cpu().numpy())
    
    val_loss /= len(test_loader)
    val_acc = val_correct / val_total
    
    # Calculate additional metrics
    val_auroc = roc_auc_score(all_targets, all_probs)
    val_precision = precision_score(all_targets, all_preds)
    val_recall = recall_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds)
    
    # Plot ROC and PR curves
    plot_roc_curve(all_targets, all_probs, epoch)
    plot_precision_recall_curve(all_targets, all_probs, epoch)
    plot_confusion_matrix(all_targets, all_preds, epoch)
    
    # Save metrics
    save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_auroc, val_precision, val_recall, val_f1)
    
    # Update learning rate based on validation performance
    scheduler.step(val_auroc)
    
    # Save model if it's the best so far
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        torch.save(model, os.path.join(output_dir, f'best_auroc_model.pth'))
        print(f"New best AUROC: {best_auroc:.4f}, model saved")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model, os.path.join(output_dir, f'best_f1_model.pth'))
        print(f"New best F1: {best_f1:.4f}, model saved")
    
    # Save the latest model checkpoint
    torch.save(model, os.path.join(output_dir, 'latest_model.pth'))
    
    # Plot training progress
    plot_training_progress()
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{args.epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
          f"Val AUROC: {val_auroc:.4f}, Val F1: {val_f1:.4f}")

print("Training completed!")
print(f"Best AUROC: {best_auroc:.4f}")
print(f"Best F1: {best_f1:.4f}")