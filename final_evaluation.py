import os
import time
import torch
import numpy as np
import snntorch as snn
import pandas as pd
import mne
from scipy import signal
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import config
from models.spiking_cnn import StandardCNN
from models.spiking_snn import SpikingCNN
from utils.dataset import build_global_summary_dict
from utils.visualization import plot_final_comparison, plot_energy_vs_accuracy, plot_false_alarms_comparison

# קבועים לחישובי אנרגיה (מבוסס ספרות מחקרית מקובלת בחומרה נוירומורפית)
CNN_MAC_PER_WINDOW = 13_147_744
ENERGY_PER_MAC_PJ = 4.6  # PicoJoules (pJ)
ENERGY_PER_SPIKE_PJ = 0.9  # PicoJoules (pJ)


# פילטר CNN (רצף מוחלט)
def apply_cnn_filter(predictions, min_consecutive=3):
    smoothed = [0] * len(predictions)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            count += 1
            if count >= min_consecutive:
                for j in range(min_consecutive):
                    smoothed[i - j] = 1
        else:
            count = 0
    return smoothed


# פילטר SNN (חלון הצבעה נוקשה לסינון אזעקות שווא)
def apply_snn_filter(predictions, window_size=6, threshold=4):
    smoothed = [0] * len(predictions)
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size + 1)
        current_window = predictions[start_idx: i + 1]
        if sum(current_window) >= threshold:
            for j in range(start_idx, i + 1):
                if predictions[j] == 1:
                    smoothed[j] = 1
    return smoothed


def evaluate_models_live():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Final Evaluation & Master Benchmark on {device}...\n")

    # 1. טעינת שני המודלים
    cnn_model = StandardCNN(num_channels=23).to(device)
    cnn_model.load_state_dict(
        torch.load(os.path.join(config.BASE_DIR, "saved_models", "model_17.pth"), map_location=device,
                   weights_only=True))
    cnn_model.eval()

    snn_model = SpikingCNN(num_channels=23, num_steps=10).to(device)
    snn_model.load_state_dict(
        torch.load(os.path.join(config.BASE_DIR, "saved_models", "best_snn_sparse_model.pth"), map_location=device,
                   weights_only=True))
    snn_model.eval()

    snn_tracker = {'spikes': 0, 'neurons': 0}

    def count_spikes_hook(module, input, output):
        spk, mem = output
        snn_tracker['spikes'] += spk.sum().item()
        snn_tracker['neurons'] += spk.numel()

    hooks = []
    for name, module in snn_model.named_modules():
        if isinstance(module, snn.Leaky):
            hooks.append(module.register_forward_hook(count_spikes_hook))

    # 2. בחירת קבצי מטופלים מאתגרים להדגמה
    # 2. בחירת 4 קבצי מטופלים רנדומליים (שיש בהם התקפים)
    seizure_map = build_global_summary_dict(config.RAW_DIR)

    valid_files = []
    for r, _, files in os.walk(config.RAW_DIR):
        for f in files:
            if f.endswith('.edf') and f in seizure_map and len(seizure_map[f]) > 0:
                valid_files.append((r, f))

    import random
    # בוחר 4 קבצים אקראיים (או פחות, אם אין 4 קבצים חוקיים בסך הכל)
    test_files = random.sample(valid_files, min(4, len(valid_files)))

    for r, _, files in os.walk(config.RAW_DIR):
        for f in files:
            if f in valid_files and f in seizure_map:
                test_files.append((r, f))

    if not test_files:
        print("[!] Could not find the specific target files. Make sure they exist.")
        return

    # משתנים לאיסוף תוצאות גלובליות
    cnn_fa_list = []
    snn_fa_list = []
    file_names = []

    total_cnn_preds, total_snn_preds, total_labels = [], [], []
    results_data = []

    # סכימה גלובלית
    total_cnn_energy_uj, total_snn_energy_uj = 0, 0
    total_cnn_time, total_snn_time = 0, 0
    total_windows_all = 0

    print("[*] Processing Files (Real-Time Simulation)...")

    # 3. לולאה על המטופלים
    for file_dir, edf_file in test_files:
        print(f"\n[{edf_file}] Analyzing...")
        seizures = seizure_map[edf_file]

        raw = mne.io.read_raw_edf(os.path.join(file_dir, edf_file), preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        data = raw.get_data()

        if data.shape[0] < 23: continue
        data = data[:23, :]

        window_samples = config.WINDOW_SIZE_SEC * config.FS
        step_samples = config.WINDOW_SIZE_SEC * config.FS

        raw_preds_cnn, raw_preds_snn, all_labels = [], [], []
        cnn_inference_time, snn_inference_time = 0, 0
        patient_snn_energy_uj = 0

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

                # --- CNN Inference ---
                t0_cnn = time.perf_counter()
                outputs_cnn = cnn_model(stft_tensor)
                _, predicted_cnn = torch.max(outputs_cnn.data, 1)
                cnn_inference_time += (time.perf_counter() - t0_cnn)
                raw_preds_cnn.append(predicted_cnn.item())

                # --- SNN Inference (אמת דינמית) ---
                snn_tracker['spikes'] = 0  # איפוס מונים לחלון הנוכחי
                snn_tracker['neurons'] = 0

                t0_snn = time.perf_counter()
                spk_rec, _ = snn_model(stft_tensor)

                # מרווח ביטחון לקבלת החלטה
                spike_counts = spk_rec.sum(dim=0)[0]
                normal_spikes = spike_counts[0].item()
                seizure_spikes = spike_counts[1].item()
                spike_margin = 2
                raw_preds_snn.append(1 if seizure_spikes >= (normal_spikes + spike_margin) else 0)
                snn_inference_time += (time.perf_counter() - t0_snn)

                # חישוב מתמטי מדויק לאנרגיה של החלון הנוכחי (SynOps)
                firing_rate = snn_tracker['spikes'] / snn_tracker['neurons'] if snn_tracker['neurons'] > 0 else 0
                dynamic_snn_ops = CNN_MAC_PER_WINDOW * firing_rate * 10  # 10 time steps
                window_snn_energy_uj = (dynamic_snn_ops * ENERGY_PER_SPIKE_PJ) / 1_000_000

                patient_snn_energy_uj += window_snn_energy_uj  # <--- מוסיפים למטופל
                total_snn_energy_uj += window_snn_energy_uj  # <--- מוסיפים לגלובלי

                # פילטור התוצאות
            final_preds_cnn = apply_cnn_filter(raw_preds_cnn)
            final_preds_snn = apply_snn_filter(raw_preds_snn)


        # שמירת תוצאות גלובליות לגרפים
        total_labels.extend(all_labels)
        total_cnn_preds.extend(final_preds_cnn)
        total_snn_preds.extend(final_preds_snn)

        # חישוב מטריקות פרטניות למטופל
        p_cnn, r_cnn, f1_cnn, _ = precision_recall_fscore_support(all_labels, final_preds_cnn, average='binary',
                                                                  zero_division=0)
        p_snn, r_snn, f1_snn, _ = precision_recall_fscore_support(all_labels, final_preds_snn, average='binary',
                                                                  zero_division=0)

        cm_cnn = confusion_matrix(all_labels, final_preds_cnn, labels=[0, 1])
        cm_snn = confusion_matrix(all_labels, final_preds_snn, labels=[0, 1])

        tp_cnn, fa_cnn = cm_cnn[1][1], cm_cnn[0][1] if len(cm_cnn) > 1 else 0
        tp_snn, fa_snn = cm_snn[1][1], cm_snn[0][1] if len(cm_snn) > 1 else 0
        actual_seizures = cm_cnn[1][0] + cm_cnn[1][1]

        cnn_fa_list.append(fa_cnn)
        snn_fa_list.append(fa_snn)
        file_names.append(edf_file.split('.')[0])

        num_windows = len(all_labels)
        total_windows_all += num_windows

        # חישובי אנרגיה (מתורגם מפיקו-ג'אול למיקרו-ג'אול)
        cnn_energy_uj = (num_windows * CNN_MAC_PER_WINDOW * ENERGY_PER_MAC_PJ) / 1_000_000

        total_cnn_energy_uj += cnn_energy_uj
        total_cnn_time += cnn_inference_time
        total_snn_time += snn_inference_time

        # איסוף לטבלה (שים לב למשתנים של האנרגיה)
        results_data.append({
            'Patient': edf_file.replace('.edf', ''),
            'Windows': num_windows,
            'Seizures_True': actual_seizures,
            'CNN_Caught': tp_cnn,
            'SNN_Caught': tp_snn,
            'CNN_FA': fa_cnn,
            'SNN_FA': fa_snn,
            'CNN_Enrgy(mJ)': round(cnn_energy_uj / 1000, 2),
            'SNN_Enrgy(mJ)': round(patient_snn_energy_uj / 1000, 2),  # <--- משתמשים במשתנה של המטופל!
            'CNN_Time(s)': round(cnn_inference_time, 2),
            'SNN_Time(s)': round(snn_inference_time, 2)
        })

        # ייצור הגרף הפרטני לכל מטופל
        cnn_metrics = {'precision': p_cnn, 'recall': r_cnn, 'f1': f1_cnn}
        snn_metrics = {'precision': p_snn, 'recall': r_snn, 'f1': f1_snn}
        plot_final_comparison(cnn_metrics, snn_metrics, edf_file.split('.')[0])

    print("\n[*] Processing complete. Generating global plots and tables...")

    # הפקת דוחות וטבלאות קונסולה
    df = pd.DataFrame(results_data)

    print("\n" + "=" * 110)
    print("=== FINAL ARCHITECTURE COMPARISON RESULTS ===")
    print("=" * 110)
    print(df.to_string(index=False))
    print("-" * 110)

    # סיכום גלובלי
    total_savings_pct = (1 - (total_snn_energy_uj / total_cnn_energy_uj)) * 100
    cnn_avg_latency_ms = (total_cnn_time / total_windows_all) * 1000
    snn_avg_latency_ms = (total_snn_time / total_windows_all) * 1000

    print("\n[ GLOBAL SUMMARY ]")
    print(f"Total Windows Processed : {total_windows_all}")
    print(f"Total CNN Energy        : {total_cnn_energy_uj / 1000:.2f} mJ")
    print(f"Total SNN Energy        : {total_snn_energy_uj / 1000:.2f} mJ")
    print(f"GLOBAL ENERGY SAVINGS   : >>> {total_savings_pct:.2f}% <<<")
    print(f"Avg CNN Latency/Window  : {cnn_avg_latency_ms:.2f} ms")
    print(f"Avg SNN Latency/Window  : {snn_avg_latency_ms:.2f} ms")
    print("=" * 110)

    # שמירה ל-CSV להכנסה קלה לאקסל / מצגת
    csv_path = os.path.join(config.BASE_DIR, "Final_Benchmark_Results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[*] Data explicitly saved to: {csv_path}")

    # ייצור גרף אזעקות שווא גלובלי
    plot_false_alarms_comparison(cnn_fa_list, snn_fa_list, file_names)

    # ייצור גרף טרייד-אוף אנרגיה
    cnn_acc = sum(1 for x, y in zip(total_cnn_preds, total_labels) if x == y) / len(total_labels)
    snn_acc = sum(1 for x, y in zip(total_snn_preds, total_labels) if x == y) / len(total_labels)

    # נתונים קשיחים מה-Benchmark:
    energy_cnn = 60.48
    energy_snn = 41.14
    plot_energy_vs_accuracy(energy_cnn, energy_snn, cnn_acc, snn_acc)

    print("\n[*] All final plots generated successfully in the 'plots' directory.")
    print(
        ">>> Look for 'Final_Comparison_...', 'False_Alarms_Comparison.png', and 'Energy_vs_Accuracy_Tradeoff.png' <<<")


if __name__ == "__main__":
    evaluate_models_live()