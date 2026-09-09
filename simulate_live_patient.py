import os
import torch
import numpy as np
import mne
from utils.visualization import save_experiment_results
from scipy import signal
from sklearn.metrics import confusion_matrix, classification_report
import random

import config
from models.spiking_cnn import StandardCNN
from utils.dataset import build_global_summary_dict


# --- הפונקציה החדשה: פילטר האזעקות ---
def apply_alarm_filter(predictions, min_consecutive=3):
    """
    מסנן אזעקות שווא: מאשר אזעקה רק אם היו X זיהויים ברצף.
    """
    smoothed = [0] * len(predictions)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            count += 1
            if count >= min_consecutive:
                # ברגע שיש רצף, אנחנו "מאשרים" את האזעקה ומחילים אותה
                # גם רטרואקטיבית על החלונות שהתחילו את הרצף
                for j in range(min_consecutive):
                    smoothed[i - j] = 1
        else:
            count = 0
    return smoothed


def run_live_simulation(model_filename="model_17.pth"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Live Patient Simulation on {device}...")

    # 1. טעינת המודל
    model = StandardCNN(num_channels=23).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", model_filename)

    if not os.path.exists(model_path):
        print(f"[!] Error: Could not find {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. איתור קובץ חולה אמיתי
    seizure_map = build_global_summary_dict(config.RAW_DIR)
    valid_files = []
    for root_dir, _, files in os.walk(config.RAW_DIR):
        for f in files:
            if f.endswith('.edf') and f in seizure_map and len(seizure_map[f]) > 0:
                valid_files.append((root_dir, f))

    if not valid_files:
        print("[!] No EDF files with seizures found.")
        return

    file_dir, edf_file = random.choice(valid_files)
    seizures = seizure_map[edf_file]
    print(f"\n[*] Selected Patient File: {edf_file}")
    print(f"[*] True Seizures in this file at seconds: {seizures}")

    # 3. סריקת הנתונים
    raw = mne.io.read_raw_edf(os.path.join(file_dir, edf_file), preload=True, verbose=False)
    raw.filter(0.5, 40.0, verbose=False)
    data = raw.get_data()

    if data.shape[0] < 23:
        print("[!] File has less than 23 channels. Skipping. Run again.")
        return
    data = data[:23, :]

    window_samples = config.WINDOW_SIZE_SEC * config.FS
    step_samples = config.WINDOW_SIZE_SEC * config.FS

    raw_preds = []
    all_labels = []

    print("\n[*] Running real-time STFT scanning (This might take a minute)...")

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

            outputs = model(stft_tensor)
            _, predicted = outputs.max(1)
            raw_preds.append(predicted.item())

    # --- 4. הפעלת הפילטר והדפסת התוצאות ---
    print("[*] Applying Debounce Filter (Min 3 consecutive windows to trigger alarm)...")
    final_preds = apply_alarm_filter(raw_preds, min_consecutive=3)

    print("\n" + "=" * 40)
    print("=== LIVE PATIENT SIMULATION RESULTS (FILTERED) ===")
    print("=" * 40)
    print(f"Total Windows Scanned: {len(all_labels)}")
    print("-" * 40)

    print(classification_report(all_labels, final_preds, target_names=['Normal', 'Seizure'], zero_division=0))

    cm = confusion_matrix(all_labels, final_preds, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    if len(cm) > 1:
        print(f"\n[!] Reality Check (After Filtering):")
        save_experiment_results(all_labels, final_preds, experiment_name="Live_Simulation_Filtered")
        print(f"-> Caught {cm[1][1]} out of {cm[1][0] + cm[1][1]} seizure windows.")
        print(f"-> False Alarms dropped to: {cm[0][1]} during the whole recording.")


if __name__ == "__main__":
    run_live_simulation("model_17.pth")