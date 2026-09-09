import os
import torch
import numpy as np
import mne
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
import random

import config
from models.spiking_snn import SpikingCNN
from utils.dataset import build_global_summary_dict
from utils.visualization import save_experiment_results


# פילטר האזעקות (Debounce)
def apply_alarm_filter(predictions, window_size=5, threshold=4):
    """
    מנגנון חלון הצבעה: מחפש X זיהויים מתוך Y חלונות אחרונים.
    עמיד בפני "פספוסים" רגעיים בתוך התקף אמיתי.
    """
    smoothed = [0] * len(predictions)
    for i in range(len(predictions)):
        # הגדרת טווח החלון (מסתכל אחורה window_size צעדים)
        start_idx = max(0, i - window_size + 1)
        current_window = predictions[start_idx: i + 1]

        # אם יש מספיק "הצבעות" התקף בחלון הנוכחי
        if sum(current_window) >= threshold:
            # מפעיל אזעקה ומחיל אותה אחורה על חלון הזמן הזה
            for j in range(start_idx, i + 1):
                # אנחנו צובעים רק את מה שהמודל באמת זיהה כדי לא למרוח סתם
                if predictions[j] == 1:
                    smoothed[j] = 1

    return smoothed


def run_live_snn_simulation(model_filename="best_snn_sparse_model.pth"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Live SNN Simulation on {device}...")

    # 1. טעינת ה-SNN
    model = SpikingCNN(num_channels=23, num_steps=10).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", model_filename)

    if not os.path.exists(model_path):
        print(f"[!] Error: Could not find {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. בחירת חולה עם התקף
    seizure_map = build_global_summary_dict(config.RAW_DIR)
    valid_files = [(r, f) for r, _, files in os.walk(config.RAW_DIR) for f in files if
                   f.endswith('.edf') and f in seizure_map and len(seizure_map[f]) > 0]

    if not valid_files:
        print("[!] No EDF files with seizures found.")
        return

    file_dir, edf_file = random.choice(valid_files)
    seizures = seizure_map[edf_file]
    print(f"[*] Patient File: {edf_file}")
    print(f"[*] True Seizures at seconds: {seizures}")

    # 3. קריאת נתונים רציפה
    raw = mne.io.read_raw_edf(os.path.join(file_dir, edf_file), preload=True, verbose=False)
    raw.filter(0.5, 40.0, verbose=False)
    data = raw.get_data()

    if data.shape[0] < 23:
        print("[!] File has less than 23 channels. Skipping.")
        return
    data = data[:23, :]

    window_samples = config.WINDOW_SIZE_SEC * config.FS
    step_samples = config.WINDOW_SIZE_SEC * config.FS

    raw_preds = []
    all_labels = []

    print("\n[*] Scanning real-time STFT (SNN inference)...")

    with torch.no_grad():
        for i in range(0, data.shape[1] - window_samples, step_samples):
            win = data[:, i: i + window_samples]
            t_start = i / config.FS
            t_end = t_start + config.WINDOW_SIZE_SEC

            is_s = any(s_s <= t_end and s_e >= t_start for (s_s, s_e) in seizures)
            all_labels.append(1 if is_s else 0)

            f, t, Zxx = signal.stft(win, fs=config.FS, nperseg=256, noverlap=128)
            stft_tensor = torch.tensor(np.abs(Zxx), dtype=torch.float32)
            stft_tensor = (stft_tensor - stft_tensor.mean()) / (stft_tensor.std() + 1e-6)
            stft_tensor = stft_tensor.unsqueeze(0).to(device)

            spk_rec, _ = model(stft_tensor)

            # סכימת ספייקים לכל מחלקה (מחלקה 0 = נורמלי, מחלקה 1 = התקף)
            # spk_rec הוא במבנה [time_steps, batch, classes]
            spike_counts = spk_rec.sum(dim=0)[0]
            normal_spikes = spike_counts[0].item()
            seizure_spikes = spike_counts[1].item()

            # הגדרת מרווח ביטחון (Spike Margin)
            spike_margin = 2

            if seizure_spikes >= (normal_spikes + spike_margin):
                raw_preds.append(1)
            else:
                raw_preds.append(0)

    # 4. הפעלת פילטר ושמירת תוצאות
    final_preds = apply_alarm_filter(raw_preds, window_size=6, threshold=4)
    print("\n" + "=" * 40)
    print("=== LIVE SNN SIMULATION RESULTS ===")
    print("=" * 40)
    print(f"Total Windows: {len(all_labels)}")
    print(classification_report(all_labels, final_preds, target_names=['Normal', 'Seizure'], zero_division=0))

    cm = confusion_matrix(all_labels, final_preds, labels=[0, 1])
    print("Confusion Matrix:\n", cm)

    if len(cm) > 1:
        print(f"\n[!] Reality Check:")
        print(f"-> Caught {cm[1][1]} out of {cm[1][0] + cm[1][1]} seizure windows.")
        print(f"-> False Alarms: {cm[0][1]}")

    # תיעוד אוטומטי
    save_experiment_results(all_labels, final_preds, experiment_name=f"Live_Simulation_SNN_{edf_file.split('.')[0]}")


if __name__ == "__main__":
    run_live_snn_simulation()