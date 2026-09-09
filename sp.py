import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. טעינת הקובץ מהנתיב המלא והמדויק
file_path = r"data/processed/test/seizure/s_4719.pt"
tensor = torch.load(file_path)

print(f"[*] Tensor shape: {tensor.shape}")

# 2. הכנת הנתונים להצגה
if tensor.dim() == 3:
    spectrogram = tensor[0].numpy()
elif tensor.dim() == 4:
    spectrogram = tensor[0, 0].numpy()
else:
    spectrogram = tensor.numpy()

# 3. ציור הספקטרוגרמה
plt.figure(figsize=(10, 5))

plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')

plt.colorbar(label='Normalized Magnitude (Z-score)')
plt.title('EEG Spectrogram - chb01_26 (Window s_4719)', fontweight='bold')
plt.ylabel('Frequency Bins', fontweight='bold')
plt.xlabel('Time Steps', fontweight='bold')

plt.tight_layout()
plt.show()