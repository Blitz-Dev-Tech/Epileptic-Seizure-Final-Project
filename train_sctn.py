import torch
import torch.nn as nn
from snntorch import functional as SF
from torch.utils.data import DataLoader
import os

import config
# הייבוא החשוב - מחליפים את ה-SpikingCNN במודל החדש שבנינו
from sctn_model import SCTNSpikingCNN
from utils.dataset import EEGProcessedDataset
from utils.visualization import save_experiment_results


def train_sctn_model():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Training SCTN Model on {device} (With Sparsity Penalty)...")

    # טעינת נתונים - בדיוק כמו בקובץ אימון ה-SNN
    train_dataset = EEGProcessedDataset(os.path.join(config.PROCESSED_DIR, "train"))
    val_dataset = EEGProcessedDataset(os.path.join(config.PROCESSED_DIR, "test"))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # יצירת מודל ה-SCTN החדש
    model = SCTNSpikingCNN(num_channels=23, num_steps=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # פונקציית Loss של snntorch שיודעת להתמודד עם ספייקים לאורך זמן
    classification_loss_fn = SF.ce_count_loss()
    sparsity_weight = 1e-5

    best_val_acc = 0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_class_loss = 0
        total_spike_penalty = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
 
            # הרצת הרשת (מחזירה ספייקים וממברנה, בדיוק כמו LIF)
            spk_rec, mem_rec = model(inputs)

            # חישוב השגיאה והקנס
            class_loss = classification_loss_fn(spk_rec, labels)
            spike_penalty = torch.sum(spk_rec) * sparsity_weight
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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc

            # שמירה בשם חדש כדי לא לדרוס את ה-SNN הרגיל שלכם
            model_path = os.path.join(config.BASE_DIR, "saved_models", "best_sctn_model.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            save_experiment_results(all_labels, all_preds, experiment_name=f"SCTN_Epoch_{epoch + 1}")

    print(f"[*] SCTN Training Complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_sctn_model()