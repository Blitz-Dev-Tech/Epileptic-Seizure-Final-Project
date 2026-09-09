import os
import time
import torch
import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config
from sctn_model import SCTNSpikingCNN, SCTN
from utils.dataset import build_global_summary_dict

# קבועים לחישוב אנרגיה אמיתית
CNN_MAC_PER_WINDOW = 13_147_744
ENERGY_PER_MAC_PJ = 4.6
ENERGY_PER_SPIKE_PJ = 0.9


def apply_snn_filter(predictions, window_size=6, threshold=4):
    """פילטר שמדמה מנגנון החלטה נוקשה כדי למנוע אזעקות שווא"""
    smoothed = [0] * len(predictions)
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size + 1)
        current_window = predictions[start_idx: i + 1]
        if sum(current_window) >= threshold:
            for j in range(start_idx, i + 1):
                if predictions[j] == 1:
                    smoothed[j] = 1
    return smoothed


def run_live_patient(edf_filename="chb01_26.edf"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n[*] ==========================================")
    print(f"[*] LIVE PATIENT SIMULATION: {edf_filename}")
    print(f"[*] ==========================================\n")

    # 1. טעינת המודל
    model_path = os.path.join(config.BASE_DIR, "saved_models", "best_sctn_model.pth")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        return

    sctn_model = SCTNSpikingCNN(num_channels=23, num_steps=10).to(device)
    sctn_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    sctn_model.eval()

    # טראקר למעקב דינמי אחרי ספייקים לחולה הנוכחי
    tracker = {'spikes': 0, 'neurons': 0}

    def count_spikes_hook(module, input, output):
        spikes, mem, timer = output
        tracker['spikes'] += spikes.sum().item()
        tracker['neurons'] += spikes.numel()

    hooks = []
    for name, module in sctn_model.named_modules():
        if isinstance(module, SCTN):
            hooks.append(module.register_forward_hook(count_spikes_hook))

    # 2. טעינת קובץ המטופל
    seizure_map = build_global_summary_dict(config.RAW_DIR)
    if edf_filename not in seizure_map:
        print(f"[!] File {edf_filename} not found in seizure map.")
        return

    file_path = None
    for root, _, files in os.walk(config.RAW_DIR):
        if edf_filename in files:
            file_path = os.path.join(root, edf_filename)
            break

    if not file_path:
        print(f"[!] Could not locate {edf_filename} on disk.")
        return

    print(f"[*] Reading raw EEG data...")
    seizures = seizure_map[edf_filename]
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(0.5, 40.0, verbose=False)
    data = raw.get_data()[:23, :] if raw.get_data().shape[0] >= 23 else raw.get_data()

    window_samples = config.WINDOW_SIZE_SEC * config.FS
    step_samples = config.WINDOW_SIZE_SEC * config.FS  # תנועה קדימה של 4 שניות

    all_labels = []
    raw_preds = []
    time_axis_sec = []

    print(f"[*] Running inference sliding window...")

    # 3. לולאת ריצת אמת
    with torch.no_grad():
        for i in range(0, data.shape[1] - window_samples, step_samples):
            win = data[:, i: i + window_samples]
            t_start = i / config.FS
            t_end = t_start + config.WINDOW_SIZE_SEC
            time_axis_sec.append(t_start)

            # תיוג אמת
            is_s = any(s_s <= t_end and s_e >= t_start for (s_s, s_e) in seizures)
            all_labels.append(1 if is_s else 0)

            # עיבוד מקדים
            f, t, Zxx = signal.stft(win, fs=config.FS, nperseg=256, noverlap=128)
            stft_tensor = torch.tensor(np.abs(Zxx), dtype=torch.float32)
            stft_tensor = ((stft_tensor - stft_tensor.mean()) / (stft_tensor.std() + 1e-6)).unsqueeze(0).to(device)

            # חיזוי
            spk_rec, _ = sctn_model(stft_tensor)
            spk_counts = spk_rec.sum(dim=0)[0]

            # סף החלטה (2 ספייקים יותר לטובת התקף)
            is_seizure_pred = 1 if spk_counts[1].item() >= (spk_counts[0].item() + 2) else 0
            raw_preds.append(is_seizure_pred)

    for h in hooks: h.remove()

    # פילטור התוצאות
    final_preds = apply_snn_filter(raw_preds)

    # 4. חישובים דינמיים מהתוצאות
    cm = confusion_matrix(all_labels, final_preds, labels=[0, 1])

    firing_rate = (tracker['spikes'] / max(1, tracker['neurons']))
    patient_sctn_ops = CNN_MAC_PER_WINDOW * firing_rate * 10 * len(all_labels)
    sctn_energy_uj = (patient_sctn_ops * ENERGY_PER_SPIKE_PJ) / 1_000_000

    cnn_energy_uj = (len(all_labels) * CNN_MAC_PER_WINDOW * ENERGY_PER_MAC_PJ) / 1_000_000
    savings = (1 - (sctn_energy_uj / max(1, cnn_energy_uj))) * 100

    print(f"\n[*] Run Complete. Generating Live Dynamic Plots...")

    # ==========================================
    # יצירת הגרפים מנתוני האמת של ההרצה
    # ==========================================
    plots_dir = os.path.join(config.BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # סגנון יפה
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    # 1. מטריצת בלבול
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'Live Confusion Matrix: {edf_filename}', fontweight='bold', fontsize=14)
    plt.ylabel('True Status', fontweight='bold')
    plt.xlabel('SCTN Prediction', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"Live_CM_{edf_filename}.png"), dpi=300)
    plt.close()

    # 2. גרף טרייד-אוף אנרגיה דינמי
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Standard CNN', 'SCTN (Live)'],
                   [cnn_energy_uj / 1_000_000, sctn_energy_uj / 1_000_000],
                   color=['#e74c3c', '#2ecc71'], width=0.5)
    plt.title(f'Live Energy Usage (mJ) for {edf_filename}', fontweight='bold', fontsize=14)
    plt.ylabel('Energy (mJ)', fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f} mJ', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.text(0.5, max(cnn_energy_uj, sctn_energy_uj) / 1_000_000 * 0.8, f"{savings:.1f}% Savings!",
             ha='center', va='center', fontsize=14, fontweight='bold', color='green',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"Live_Energy_{edf_filename}.png"), dpi=300)
    plt.close()

    # 3. גרף ציר זמן התקפים (The Money Shot)
    plt.figure(figsize=(12, 4))

    # המרת נתונים כדי לצייר יפה
    time_mins = np.array(time_axis_sec) / 60.0

    # ציור האמת כשטח אדום
    plt.fill_between(time_mins, 0, all_labels, color='#ff9999', alpha=0.5, label='Actual Seizure', step='post')

    # ציור חיזוי המודל כקו כחול
    plt.step(time_mins, final_preds, color='#2c3e50', linewidth=2.5, where='post', label='SCTN Alarm')

    plt.title(f'Real-Time Seizure Detection Timeline: {edf_filename}', fontweight='bold', fontsize=14)
    plt.xlabel('Time (Minutes)', fontweight='bold')
    plt.ylabel('Alarm Status', fontweight='bold')
    plt.yticks([0, 1], ['Normal', 'Alarm!'], fontweight='bold')
    plt.legend(loc='upper right')
    plt.ylim(-0.1, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"Live_Timeline_{edf_filename}.png"), dpi=300)
    plt.close()

    print(f"[*] Done! 3 Real-time graphs generated in the 'plots' folder.")
    print(f"    -> Look for 'Live_Timeline_{edf_filename}.png'")


if __name__ == "__main__":
    # אתם יכולים לשנות פה את שם הקובץ לכל חולה אחר כדי לייצר לו גרפים חיים
    run_live_patient("chb01_26.edf")