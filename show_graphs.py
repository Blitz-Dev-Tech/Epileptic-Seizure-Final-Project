import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# הנתונים המטורפים אחרי הפילטר (Debounce)
cm = np.array([[880, 3], [5, 11]])

metrics_data = {
    'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
    'Class': ['Normal'] * 3 + ['Seizure'] * 3,
    'Value': [0.99, 1.00, 1.00, 0.79, 0.69, 0.73]
}
df_metrics = pd.DataFrame(metrics_data)

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)
sns.set_theme(style="whitegrid")

# --- גרף 1: מטריצת בלבול (אחרי סינון) ---
plt.figure(figsize=(8, 6))
ax_cm = sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Normal (Predicted)', 'Seizure (Alarm)'],
                    yticklabels=['Normal (Actual)', 'Seizure (Actual)'],
                    annot_kws={"size": 15})
plt.title('Confusion Matrix - FILTERED Live Patient (~60 mins)', fontsize=14, pad=15)
plt.yticks(rotation=0)
plt.savefig(os.path.join(plots_dir, "filtered_live_cm.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- גרף 2: מדדי ביצועים ---
plt.figure(figsize=(10, 6))
ax_metrics = sns.barplot(data=df_metrics, x='Metric', y='Value', hue='Class', palette='Set2')
plt.title('Performance Metrics - Filtered System', fontsize=15, pad=15)
plt.ylim(0, 1.1)

for p in ax_metrics.patches:
    ax_metrics.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points', fontsize=12)

plt.savefig(os.path.join(plots_dir, "filtered_live_metrics.png"), dpi=300, bbox_inches='tight')
plt.close()
print("[*] Boom! Presentation-ready graphs saved in 'plots' folder.")