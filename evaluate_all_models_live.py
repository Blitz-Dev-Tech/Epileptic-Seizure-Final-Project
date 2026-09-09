import os
import time
import torch
import numpy as np
import snntorch as snn
import pandas as pd
import mne
from scipy import signal
from sklearn.metrics import confusion_matrix

import config
from models.spiking_cnn import StandardCNN
from models.spiking_snn import SpikingCNN
from sctn_model import SCTNSpikingCNN, SCTN
from utils.dataset import build_global_summary_dict

CNN_MAC_PER_WINDOW = 13_147_744
ENERGY_PER_MAC_PJ = 4.6
ENERGY_PER_SPIKE_PJ = 0.9


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


def evaluate_all_models_live():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Ultimate 3-Way Evaluation on {device}...\n")

    # 1. טעינת כל המודלים
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

    sctn_model = SCTNSpikingCNN(num_channels=23, num_steps=10).to(device)
    sctn_model.load_state_dict(
        torch.load(os.path.join(config.BASE_DIR, "saved_models", "best_sctn_model.pth"), map_location=device,
                   weights_only=True))
    sctn_model.eval()

    # טראקרים לספייקים
    snn_tracker = {'spikes': 0, 'neurons': 0}
    sctn_tracker = {'spikes': 0, 'neurons': 0}

    def count_snn_spikes_hook(module, input, output):
        spk, mem = output
        snn_tracker['spikes'] += spk.sum().item()
        snn_tracker['neurons'] += spk.numel()

    def count_sctn_spikes_hook(module, input, output):
        spikes, mem, timer = output
        sctn_tracker['spikes'] += spikes.sum().item()
        sctn_tracker['neurons'] += spikes.numel()

    # חיבור Hooks
    for name, module in snn_model.named_modules():
        if isinstance(module, snn.Leaky): module.register_forward_hook(count_snn_spikes_hook)
    for name, module in sctn_model.named_modules():
        if isinstance(module, SCTN): module.register_forward_hook(count_sctn_spikes_hook)

    # הכנת קבצים
    seizure_map = build_global_summary_dict(config.RAW_DIR)
    valid_files = [(r, f) for r, _, files in os.walk(config.RAW_DIR) for f in files if
                   f.endswith('.edf') and f in seizure_map and len(seizure_map[f]) > 0]

    import random
    test_files = random.sample(valid_files, min(4, len(valid_files)))

    results_data = []

    print("[*] Processing Files (Comparing Architecture Energetics)...")

    for file_dir, edf_file in test_files:
        seizures = seizure_map[edf_file]
        raw = mne.io.read_raw_edf(os.path.join(file_dir, edf_file), preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        data = raw.get_data()[:23, :] if raw.get_data().shape[0] >= 23 else raw.get_data()

        window_samples = config.WINDOW_SIZE_SEC * config.FS
        step_samples = config.WINDOW_SIZE_SEC * config.FS

        all_labels = []
        raw_preds_cnn, raw_preds_snn, raw_preds_sctn = [], [], []
        patient_snn_energy, patient_sctn_energy = 0, 0

        with torch.no_grad():
            for i in range(0, data.shape[1] - window_samples, step_samples):
                win = data[:, i: i + window_samples]
                t_end = (i / config.FS) + config.WINDOW_SIZE_SEC
                is_s = any(s_s <= t_end and s_e >= (i / config.FS) for (s_s, s_e) in seizures)
                all_labels.append(1 if is_s else 0)

                f, t, Zxx = signal.stft(win, fs=config.FS, nperseg=256, noverlap=128)
                stft_tensor = torch.tensor(np.abs(Zxx), dtype=torch.float32)
                stft_tensor = ((stft_tensor - stft_tensor.mean()) / (stft_tensor.std() + 1e-6)).unsqueeze(0).to(device)

                # CNN
                _, predicted_cnn = torch.max(cnn_model(stft_tensor).data, 1)
                raw_preds_cnn.append(predicted_cnn.item())

                # SNN
                snn_tracker['spikes'] = 0;
                snn_tracker['neurons'] = 0
                spk_rec_snn, _ = snn_model(stft_tensor)
                snn_spk_counts = spk_rec_snn.sum(dim=0)[0]
                raw_preds_snn.append(1 if snn_spk_counts[1].item() >= (snn_spk_counts[0].item() + 2) else 0)
                patient_snn_energy += ((CNN_MAC_PER_WINDOW * (snn_tracker['spikes'] / max(1, snn_tracker[
                    'neurons'])) * 10) * ENERGY_PER_SPIKE_PJ) / 1_000_000

                # SCTN
                sctn_tracker['spikes'] = 0;
                sctn_tracker['neurons'] = 0
                spk_rec_sctn, _ = sctn_model(stft_tensor)
                sctn_spk_counts = spk_rec_sctn.sum(dim=0)[0]
                raw_preds_sctn.append(1 if sctn_spk_counts[1].item() >= (sctn_spk_counts[0].item() + 2) else 0)
                patient_sctn_energy += ((CNN_MAC_PER_WINDOW * (sctn_tracker['spikes'] / max(1, sctn_tracker[
                    'neurons'])) * 10) * ENERGY_PER_SPIKE_PJ) / 1_000_000

        # פילטור זהה לשלושתם
        final_cnn = apply_cnn_filter(raw_preds_cnn)
        final_snn = apply_snn_filter(raw_preds_snn)
        final_sctn = apply_snn_filter(raw_preds_sctn)

        # חישוב מטריקות בסיסיות למטופל
        cm_cnn = confusion_matrix(all_labels, final_cnn, labels=[0, 1])
        cm_snn = confusion_matrix(all_labels, final_snn, labels=[0, 1])
        cm_sctn = confusion_matrix(all_labels, final_sctn, labels=[0, 1])

        cnn_energy_uj = (len(all_labels) * CNN_MAC_PER_WINDOW * ENERGY_PER_MAC_PJ) / 1_000_000

        results_data.append({
            'Patient': edf_file.replace('.edf', ''),
            'Seizures': cm_cnn[1][0] + cm_cnn[1][1],
            'CNN_Caught': cm_cnn[1][1], 'CNN_FA': cm_cnn[0][1], 'CNN_mJ': round(cnn_energy_uj / 1000, 2),
            'SNN_Caught': cm_snn[1][1], 'SNN_FA': cm_snn[0][1], 'SNN_mJ': round(patient_snn_energy / 1000, 2),
            'SCTN_Caught': cm_sctn[1][1], 'SCTN_FA': cm_sctn[0][1], 'SCTN_mJ': round(patient_sctn_energy / 1000, 2)
        })

    df = pd.DataFrame(results_data)
    print("\n" + "=" * 110)
    print("=== FINAL 3-WAY ARCHITECTURE SHOWDOWN ===")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    evaluate_all_models_live()