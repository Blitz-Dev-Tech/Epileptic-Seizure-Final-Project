import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from torch.utils.data import DataLoader
import os

import config
from models.spiking_snn import SpikingCNN
from utils.dataset import EEGProcessedDataset
from utils.visualization import save_experiment_results


def train_snn_with_sparsity():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Training SNN on {device} (With Sparsity Penalty)...")

    train_dataset = EEGProcessedDataset(os.path.join(config.PROCESSED_DIR, "train"))
    val_dataset = EEGProcessedDataset(os.path.join(config.PROCESSED_DIR, "test"))  # תיקון ל-test

    # הוספת num_workers ו-pin_memory לשיפור ביצועי GPU
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SpikingCNN(num_channels=23, num_steps=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # הפסד ראשי: דיוק סיווג
    classification_loss_fn = SF.ce_count_loss()

    # מקדם הקנס על ספייקים (Hyperparameter) - נתחיל קטן כדי לא "להרוג" את הלמידה
    sparsity_weight = 1e-4

    best_val_acc = 0

    for epoch in range(config.EPOCHS):  # תיקון ל-EPOCHS בהנחה שזה מה שיש ב-config
        model.train()
        total_loss = 0
        total_class_loss = 0
        total_spike_penalty = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            spk_rec, mem_rec = model(inputs)

            # 1. חישוב שגיאת הסיווג
            class_loss = classification_loss_fn(spk_rec, labels)

            # 2. חישוב קנס הדלילות (L1 Penalty על ספייקים)
            # אנחנו סוכמים את כל הספייקים (כמה שיותר ספייקים = עונש גדול יותר)
            spike_penalty = torch.sum(spk_rec) * sparsity_weight

            # השגיאה הכוללת היא השילוב של שניהם
            loss_val = class_loss + spike_penalty

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()
            total_class_loss += class_loss.item()
            total_spike_penalty += spike_penalty.item()

            if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                print(
                    f"   [Epoch {epoch + 1} | Batch {i + 1}/{len(train_loader)}] Loss: {loss_val.item():.4f} (Class: {class_loss.item():.4f}, Penalty: {spike_penalty.item():.4f})")

        # Validation
        model.eval()
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                spk_rec, _ = model(inputs)

                _, predicted = spk_rec.sum(dim=0).max(1)

                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / len(val_dataset)
        avg_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_penalty = total_spike_penalty / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] - Total Loss: {avg_loss:.4f} (Class: {avg_class_loss:.4f}, Pen: {avg_penalty:.4f}) - Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:  # שומר גם אם שווה, כדי לתת עדיפות למודל מאוחר יותר עם פוטנציאל דלילות גבוה יותר
            best_val_acc = val_acc
            model_path = os.path.join(config.BASE_DIR, "saved_models", "best_snn_sparse_model.pth")  # שם חדש
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            save_experiment_results(all_labels, all_preds, experiment_name=f"SNN_Sparse_Epoch_{epoch + 1}")

    print(f"[*] Sparse Training Complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_snn_with_sparsity()