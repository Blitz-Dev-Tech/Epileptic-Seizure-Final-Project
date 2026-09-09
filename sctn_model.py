import torch
import torch.nn as nn
from snntorch import surrogate


class SCTN(nn.Module):
    def __init__(self, theta=0.0, leakage_factor=0, leakage_period=5, threshold=0.15, reset_to=0.0, spike_grad=None):
        super().__init__()
        self.theta = theta
        self.leakage_factor = leakage_factor
        self.leakage_period = leakage_period
        self.threshold = threshold
        self.reset_to = reset_to

        # אנחנו חייבים להשתמש בגרדיאנט כדי לאמן רשת SNN!
        self.spike_grad = spike_grad if spike_grad is not None else surrogate.fast_sigmoid(slope=25)

    def init_sctn(self):
        # מחזיר (מתח, טיימר) כ-None. הם יקבלו את הגודל המדויק אוטומטית כשהדאטה ייכנס
        return None, None

    def forward(self, current, mem, timer):
        # אתחול אוטומטי של הגדלים לפי התמונות (Batch Size)
        if mem is None: mem = torch.zeros_like(current)
        if timer is None: timer = torch.zeros_like(current)

        # 1. חישוב הצטברות המתח לפי המשוואות של ה-SCTN
        if self.leakage_factor < 3:
            mem = mem + current + self.theta
        else:
            lf_mult = 2 ** (self.leakage_factor - 3)
            mem = mem + (current + self.theta) * lf_mult

        # 2. יצירת ספייק בעזרת הגרדיאנט
        spikes = self.spike_grad(mem - self.threshold)

        # 3. מנגנון הדליפה (Leakage Period)
        timer = timer + 1
        mask = (timer >= self.leakage_period).float()  # האם עבר הזמן לדליפה?

        decay_delta = torch.where(
            mem < 0,
            (-mem) / (2 ** self.leakage_factor),
            -(mem / (2 ** self.leakage_factor))
        )

        # עדכון המתח והטיימר רק אם ה-mask הוא 1 (הזמן עבר)
        mem = mem + (decay_delta * mask)
        timer = timer * (1 - mask)

        # 4. איפוס המתח (Reset) אם הנוירון ירה ספייק
        mem = torch.where(spikes > 0, torch.full_like(mem, self.reset_to), mem)

        return spikes, mem, timer


class SCTNSpikingCNN(nn.Module):
    def __init__(self, num_channels=23, num_steps=10):
        super().__init__()

        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # שכבת קונבולוציה 1
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.sctn1 = SCTN(spike_grad=spike_grad, leakage_factor=1, leakage_period=3, threshold=1.0)
        self.pool1 = nn.MaxPool2d(2)

        # שכבת קונבולוציה 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.sctn2 = SCTN(spike_grad=spike_grad, leakage_factor=1, leakage_period=3, threshold=1.0)
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        # שכבות סיווג (Fully Connected)
        self.fc1 = nn.Linear(64 * 32 * 2, 128)
        self.sctn3 = SCTN(spike_grad=spike_grad, leakage_factor=1, leakage_period=3, threshold=1.0)

        self.fc2 = nn.Linear(128, 2)
        # שכבת הפלט
        self.sctn4 = SCTN(spike_grad=spike_grad, leakage_factor=1, leakage_period=3, threshold=1.0)

    def forward(self, x):
        # 1. איפוס המתח והטיימר של כל הנוירונים בתחילת החלון (מקבלים mem ו-timer)
        mem1, timer1 = self.sctn1.init_sctn()
        mem2, timer2 = self.sctn2.init_sctn()
        mem3, timer3 = self.sctn3.init_sctn()
        mem4, timer4 = self.sctn4.init_sctn()

        spk4_rec = []
        mem4_rec = []

        # 2. לולאת הזמן
        for step in range(self.num_steps):
            cur1 = self.pool1(self.conv1(x))
            spk1, mem1, timer1 = self.sctn1(cur1, mem1, timer1)

            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2, timer2 = self.sctn2(cur2, mem2, timer2)

            cur3 = self.fc1(self.flatten(spk2))
            spk3, mem3, timer3 = self.sctn3(cur3, mem3, timer3)

            cur4 = self.fc2(spk3)
            spk4, mem4, timer4 = self.sctn4(cur4, mem4, timer4)

            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)