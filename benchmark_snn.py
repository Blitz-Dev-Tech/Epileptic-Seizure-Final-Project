import torch
import time
import os
import snntorch as snn

import config
from models.spiking_snn import SpikingCNN


def run_snn_benchmark(model_filename="best_snn_model.pth"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Running SNN Benchmark on {device}...\n")

    # 1. טעינת המודל
    model = SpikingCNN(num_channels=23, num_steps=10).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", model_filename)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"[*] Loaded weights from {model_filename}")
    else:
        print(f"[!] Error: {model_filename} not found. Train the model first.")
        return

    model.eval()

    # 2. נתון דמה (Dummy Input) לחלון של 2 שניות
    dummy_input = torch.randn(1, 23, 129, 9).to(device)

    # ==========================================
    # מדד 1: גודל המודל (פרמטרים)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters())

    # ==========================================
    # מדד 2: כמות ספייקים (Sparsity)
    # ==========================================
    tracker = {'spikes': 0, 'neurons_checked': 0}

    # פונקציית Hook שסופרת ספייקים בכל שכבה ביולוגית
    def count_spikes_hook(module, input, output):
        spk, mem = output
        tracker['spikes'] += spk.sum().item()
        tracker['neurons_checked'] += spk.numel()

    # חיבור ה-Hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, snn.Leaky):
            hooks.append(module.register_forward_hook(count_spikes_hook))

    # הרצה אחת קפואה לספירת ספייקים
    with torch.no_grad():
        _ = model(dummy_input)

    # הסרת ה-Hooks כדי לא לפגוע במדידת המהירות
    for h in hooks:
        h.remove()

    total_spikes = tracker['spikes']
    firing_rate = (total_spikes / tracker['neurons_checked']) * 100 if tracker['neurons_checked'] > 0 else 0

    # ==========================================
    # מדד 3: זמן הסקה (Latency)
    # ==========================================
    for _ in range(10): _ = model(dummy_input)  # חימום

    runs = 100
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    avg_latency_ms = ((end_time - start_time) / runs) * 1000

    # ==========================================
    # הדפסת התוצאות והערכת האנרגיה
    # ==========================================
    print("\n" + "=" * 45)
    print("=== SNN ARCHITECTURE BENCHMARK ===")
    print("=" * 45)
    print(f"1. Model Parameters : {total_params:,}")
    print(f"2. Total Spikes     : {int(total_spikes):,} per window")
    print(f"3. Avg Firing Rate  : {firing_rate:.2f}% (Sparsity: {100 - firing_rate:.2f}%)")
    print(f"4. Avg Latency      : {avg_latency_ms:.3f} ms per window")
    print("-" * 45)

    # חישוב מתמטי מקובל של צריכת אנרגיה (Neuromorphic vs Standard)
    cnn_macs = 13_147_744  # מה-Benchmark הקודם
    snn_ops = cnn_macs * (firing_rate / 100) * 10  # 10 Time steps

    energy_mac = 4.6  # פיקו-ג'אול (pJ) לפעולת כפל-חיבור
    energy_ac = 0.9  # פיקו-ג'אול (pJ) לפעולת חיבור בלבד (ספייק)

    cnn_energy = cnn_macs * energy_mac
    snn_energy = snn_ops * energy_ac

    print("\n[!] THEORETICAL ENERGY ESTIMATION [!]")
    print(f"CNN Energy per window: ~{cnn_energy / 1_000_000:.2f} Micro-Joules (uJ)")
    print(f"SNN Energy per window: ~{snn_energy / 1_000_000:.2f} Micro-Joules (uJ)")

    if cnn_energy > 0:
        savings = (1 - (snn_energy / cnn_energy)) * 100
        print(f">>> Estimated Energy Savings vs CNN: {savings:.2f}% <<<")
    print("=" * 45)


if __name__ == "__main__":
    run_snn_benchmark("best_snn_sparse_model.pth")