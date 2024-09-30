import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, accuracy_score)
import seaborn as sns

from utils import Find_Optimal_Cutoff, ensure_dir



# Evaluation function
def evaluate_model(model, val_loader, criterion, model_dir, use_static_threshold=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    classes = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
               'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    # Ensure 'results' directory exists
    results_dir = os.path.join(model_dir, 'results')
    ensure_dir(results_dir)

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

    val_loss /= len(val_loader)
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Threshold outputs using either a static threshold or optimal cutoff per class
    all_preds = np.zeros_like(all_outputs)
    
    for i in range(all_labels.shape[1]):
        if use_static_threshold:
            threshold = 0.5
        else:
            threshold = Find_Optimal_Cutoff(all_labels[:, i], all_outputs[:, i])
        all_preds[:, i] = (all_outputs[:, i] > threshold).astype(int)

    # Plot ROC Curves
    plt.figure(figsize=(10, 8))
    auc_scores = []

    for i, class_name in enumerate(classes):
        if np.unique(all_labels[:, i]).size > 1:  # Check for both classes in the label
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            auc_scores.append(auc)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')

    plt.title('ROC Curves per Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()

    # Confusion Matrices
    for i, class_name in enumerate(classes):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f'Confusion Matrix for {class_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_{class_name}.png'))
        plt.close()

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

    # Calculate averaged metrics
    accuracy_per_class = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    mean_accuracy = np.mean(accuracy_per_class)
    mean_auc = np.nanmean(auc_scores)  # Averaged AUROC

    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    # Output classification report per class and overall metrics
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Average Accuracy per Class: {mean_accuracy:.4f}')
    print(f'Mean AUROC: {mean_auc:.4f}')
    print(f'Micro-Averaged Precision: {precision_micro:.4f}')
    print(f'Micro-Averaged Recall: {recall_micro:.4f}')
    print(f'Micro-Averaged F1-Score: {f1_micro:.4f}')

    # Save full classification report to file
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Validation Loss: {val_loss:.4f}\n')
        f.write(f'Average Accuracy per Class: {mean_accuracy:.4f}\n')
        f.write(f'Mean AUROC: {mean_auc:.4f}\n')
        f.write(f'Micro-Averaged Precision: {precision_micro:.4f}\n')
        f.write(f'Micro-Averaged Recall: {recall_micro:.4f}\n')
        f.write(f'Micro-Averaged F1-Score: {f1_micro:.4f}\n\n')
        f.write('Classification Report per Class:\n')
        for class_name, metrics in report.items():
            f.write(f'{class_name}:\n')
            for metric_name, metric_value in metrics.items():
                f.write(f'  {metric_name}: {metric_value:.4f}\n')

    return val_loss, mean_auc, precision_micro, recall_micro, f1_micro
