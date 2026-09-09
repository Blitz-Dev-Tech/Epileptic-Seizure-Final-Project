import os
import copy
from evaluate import evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ייבוא מהקבצים שלנו
import config
from utils.dataset import EEGProcessedDataset
from models.spiking_cnn import StandardCNN


def main():
    # 1. הגדרת מכשיר (GPU אם יש, אחרת CPU)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 2. טעינת הנתונים
    print("[*] Loading processed datasets...")
    train_dir = os.path.join(config.PROCESSED_DIR, "train")
    test_dir = os.path.join(config.PROCESSED_DIR, "test")

    train_dataset = EEGProcessedDataset(train_dir)
    test_dataset = EEGProcessedDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. יצירת המודל
    model = StandardCNN(num_channels=23).to(device)

    criterion = nn.CrossEntropyLoss()

    # --- תוספת 1: קנסות (Weight Decay) ---
    # ההגדרה weight_decay=1e-4 אומרת למודל למנוע משקולות קיצוניות
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # --- תוספת 2: הגדרות עצירה אוטומטית (Early Stopping) ---
    patience = 10  # כמה אפוקים לחכות בלי שיפור לפני שעוצרים
    epochs_no_improve = 0
    best_test_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())  # שומר את המצב ההתחלתי

    # 4. לופ האימון
    print("[*] Starting Training Loop with Early Stopping & Penalties...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # שלב אימון (Train)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total if total > 0 else 0
        train_loss = running_loss / len(train_loader)

        # שלב בחינה (Test)
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # אנחנו עוקבים אחרי ה-Loss של הטסט, לא רק אחרי האחוזים
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total if test_total > 0 else 0
        test_loss = test_running_loss / len(test_loader)

        print(
            f"=== Epoch {epoch + 1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f} ===")

        # --- הפעלת העצירה האוטומטית ---
        # אנחנו רוצים שה-Loss (השגיאה) ירד. אם הוא הכי נמוך שהיה לנו - שומרים!
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"    [*] New Best Model Saved! (Loss dropped to {best_test_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    [!] No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"\n[!!!] Early stopping triggered after {epoch + 1} epochs. Stopping training.")
            break  # עוצר את הלופ הראשי

    # 5. סיום ושמירה סופית
    # טוענים חזרה את המשקולות הכי טובות שמצאנו לפני שהתחיל ה-Overfitting
    model.load_state_dict(best_model_wts)
    print("\n[*] Training Complete. Best model restored.")

    # שומרים את המודל הסופי לדיסק כדי שנוכל להשתמש בו אחר כך
    torch.save(model.state_dict(), os.path.join(config.BASE_DIR, "saved_models/model_17.pth"))
    print("[*] Automatically evaluating the best model...")
    evaluate_model("best_epilepsy_model.pth")


if __name__ == "__main__":
    main()