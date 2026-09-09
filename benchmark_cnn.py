import torch
import time
import os
from thop import profile
from thop import clever_format

import config
from models.spiking_cnn import StandardCNN


def run_benchmark(model_filename="model_17.pth"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Running CNN Benchmark on {device}...\n")

    # 1. טעינת המודל
    model = StandardCNN(num_channels=23).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", model_filename)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"[*] Loaded weights from {model_filename}")
    else:
        print(f"[!] Warning: {model_filename} not found. Running benchmark on an untrained model (architecture only).")

    model.eval()  # חובה למדידות זמן מדויקות

    # 2. יצירת נתון דמה (Dummy Input) המדמה חלון של 2 שניות
    # המבנה: [Batch=1, Channels=23, Freq=129, Time=9]
    dummy_input = torch.randn(1, 23, 129, 9).to(device)

    # ==========================================
    # מדד 1: גודל המודל (Model Size / Parameters)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ==========================================
    # מדד 2: כמות פעולות מתמטיות (Compute / MACs)
    # ==========================================
    # משתמשים בספריית thop כדי לחשב כמה חישובים קורים בחלון אחד
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.2f")

    # ==========================================
    # מדד 3: זמן הסקה חלון בודד (Inference Latency)
    # ==========================================
    # "חימום" המעבד/GPU כדי לקבל מדידה אמיתית
    for _ in range(10):
        _ = model(dummy_input)

    runs = 100
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / runs) * 1000

    # ==========================================
    # הדפסת דוח ה-Benchmark
    # ==========================================
    print("\n" + "=" * 40)
    print("=== CNN ARCHITECTURE BENCHMARK ===")
    print("=" * 40)
    print(f"1. Model Parameters : {total_params:,} (Trainable: {trainable_params:,})")
    print(f"2. Compute (MACs)   : {macs:,} operations per window ({macs_formatted})")
    print(f"3. Avg Latency      : {avg_latency_ms:.3f} ms per window")
    print("=" * 40)
    print("Note: 1 MAC (Multiply-Accumulate) is considered as 2 FLOPs in some literature.")


if __name__ == "__main__":
    run_benchmark("model_17.pth")