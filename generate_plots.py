import matplotlib.pyplot as plt
import numpy as np

# הגדרות סגנון נקי ומרשים למצגות
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1'] # אדום, ירוק-מים, כחול

# הנתונים המסוכמים מהריצה שלכם
models = ['CNN (Baseline)', 'SNN (LIF)', 'SCTN (New)']
accuracy = [81.25, 72.50, 55.00]
false_alarms = [60, 56, 0]
energy = [198.25, 18.36, 16.49]

def autolabel(rects, ax, suffix=''):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# ==========================================
# 1. גרף צריכת אנרגיה (החיסכון הענק)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(models, energy, color=colors, width=0.6)
ax.set_ylabel('Total Energy Consumption (mJ)', fontsize=12, fontweight='bold')
ax.set_title('Energy Consumption Comparison (Lower is Better)', fontsize=14, fontweight='bold')
autolabel(bars, ax, ' mJ')
plt.tight_layout()
plt.savefig('Plot_1_Energy.png', dpi=300)

# ==========================================
# 2. גרף זיהוי התקפים לעומת אזעקות שווא (הטרייד-אוף)
# ==========================================
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy, width, label='Detection Rate (%)', color='#2ecc71')
rects2 = ax.bar(x + width/2, false_alarms, width, label='False Alarms (Count)', color='#e74c3c')

ax.set_ylabel('Score / Count', fontsize=12, fontweight='bold')
ax.set_title('Detection Rate vs. False Alarms', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.legend()

autolabel(rects1, ax, '%')
autolabel(rects2, ax)
plt.tight_layout()
plt.savefig('Plot_2_Accuracy_vs_FA.png', dpi=300)

# ==========================================
# 3. ניתוח רמת מטופל (אנרגיה לכל פציינט)
# ==========================================
patients = ['chb18_32', 'chb22_38', 'chb01_26', 'chb20_13']
cnn_patient_energy = [54.37, 54.37, 35.14, 54.37]
snn_patient_energy = [5.41, 6.19, 3.08, 3.68]
sctn_patient_energy = [5.16, 5.54, 2.80, 2.99]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(patients, cnn_patient_energy, marker='o', linewidth=3, label='CNN', color=colors[0])
ax.plot(patients, snn_patient_energy, marker='s', linewidth=3, label='SNN (LIF)', color=colors[1])
ax.plot(patients, sctn_patient_energy, marker='^', linewidth=3, label='SCTN', color=colors[2])

ax.set_ylabel('Energy (mJ)', fontsize=12, fontweight='bold')
ax.set_title('Energy Consumption per Patient', fontsize=14, fontweight='bold')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Plot_3_Patient_Energy.png', dpi=300)

print("[*] Successfully generated 3 high-resolution graphs for the presentation!")