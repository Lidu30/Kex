import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from datasets import load_schizophrenia_data, preprocess_schizophrenia_data
from torch.utils.data import TensorDataset, DataLoader

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate Schizophrenia EEG Classification Model')
parser.add_argument('--healthy_dir', type=str, required=True,
                    help='Path to the directory containing healthy subject data')
parser.add_argument('--schizophrenia_dir', type=str, required=True,
                    help='Path to the directory containing schizophrenia subject data')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the trained model')
parser.add_argument('--output_dir', type=str, default='./Results/Evaluation/',
                    help='Path to save evaluation results')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
print(f"Loading model from {args.model_path}")
model = torch.load(args.model_path, map_location=device)
model.eval()

# Load data
print("Loading and preprocessing data...")
X_raw, y = load_schizophrenia_data(args.healthy_dir, args.schizophrenia_dir)
X_processed = preprocess_schizophrenia_data(X_raw)

# Reshape for the KAN model (flatten the 3D features)
n_samples = X_processed.shape[0]
X_reshaped = X_processed.reshape(n_samples, -1)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_reshaped)
y_tensor = torch.tensor(y, dtype=torch.int64)

# Create dataset and dataloader
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Evaluate the model
print("Evaluating model...")
all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        
        # Get predictions
        _, predicted = output.max(1)
        
        # Store predictions and targets
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(output.softmax(dim=1)[:, 1].cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# Calculate metrics
accuracy = (all_preds == all_targets).mean()
auroc = roc_auc_score(all_targets, all_probs)
precision = precision_score(all_targets, all_preds)
recall = recall_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
report = classification_report(all_targets, all_preds, target_names=['Healthy', 'Schizophrenia'])

# Print metrics
print("\n===== Evaluation Results =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save metrics to file
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    f.write("===== Evaluation Results =====\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"AUROC: {auroc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n\n")
    f.write("Classification Report:\n")
    f.write(f"{report}\n")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
fpr, tpr, _ = roc_curve(all_targets, all_probs)
plt.plot(fpr, tpr, label=f'ROC (AUC = {auroc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()

# Plot Precision-Recall curve
plt.subplot(2, 1, 2)
precision_values, recall_values, _ = precision_recall_curve(all_targets, all_probs)
average_precision = average_precision_score(all_targets, all_probs)
plt.plot(recall_values, precision_values, label=f'PR (AP = {average_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
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
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

print(f"\nEvaluation results saved to {output_dir}")