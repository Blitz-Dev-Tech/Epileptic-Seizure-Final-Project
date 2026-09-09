import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.visualization import save_experiment_results
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader

import config
from utils.dataset import EEGProcessedDataset
from models.spiking_cnn import StandardCNN


def plot_and_save_metrics(all_labels, all_preds, title_suffix=""):
    """
    פונקציה חכמה שלוקחת את התוצאות האמיתיות ומייצרת מהן גרפים
    """
    # 1. חישוב אוטומטי של מטריצת הבלבול
    cm = confusion_matrix(all_labels, all_preds)

    # 2. חישוב אוטומטי של מדדי הביצועים
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1])

    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
        'Class': ['Normal'] * 3 + ['Seizure'] * 3,
        'Value': [precision[0], recall[0], f1[0], precision[1], recall[1], f1[1]]
    }
    df_metrics = pd.DataFrame(metrics_data)

    # הגדרת סביבת שמירה
    plots_dir = os.path.join(config.BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- יצירת גרף 1: מטריצת בלבול ---
    plt.figure(figsize=(8, 6))
    ax_cm = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal (Pred)', 'Seizure (Pred)'],
                        yticklabels=['Normal (True)', 'Seizure (True)'],
                        annot_kws={"size": 15})
    plt.title(f'Confusion Matrix {title_suffix}', fontsize=16, pad=15)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()  # מקפיץ את הגרף למסך

    # --- יצירת גרף 2: מדדי ביצועים ---
    plt.figure(figsize=(10, 6))
    ax_metrics = sns.barplot(data=df_metrics, x='Metric', y='Value', hue='Class', palette='muted')
    plt.title(f'Performance Metrics {title_suffix}', fontsize=16, pad=15)
    plt.ylim(0, 1.0)

    for p in ax_metrics.patches:
        ax_metrics.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9),
                            textcoords='offset points', fontsize=12)

    plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()  # מקפיץ את הגרף למסך


def evaluate_model(model_filename="model_17.pth"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    test_dir = os.path.join(config.PROCESSED_DIR, "test")
    test_dataset = EEGProcessedDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = StandardCNN(num_channels=23).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", model_filename)

    if not os.path.exists(model_path):
        print(f"[!] Error: Could not find {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"[*] Evaluating {model_filename} on {len(test_dataset)} windows...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # הדפסה לטרמינל
    print("\n=== Real Test Results ===")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Seizure']))
    save_experiment_results(all_labels, all_preds, experiment_name="Standard_Eval_Model17")

    # קריאה לפונקציה האוטומטית שמציירת את הגרפים!
    plot_and_save_metrics(all_labels, all_preds, title_suffix=f"({model_filename})")


if __name__ == "__main__":
    # אפשר לשנות כאן את השם לכל מודל אחר שתשמור בעתיד
    evaluate_model("model_17.pth")