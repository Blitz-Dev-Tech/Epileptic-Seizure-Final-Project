import torch
import torch.nn as nn


class SCTNeuronPyTorch(nn.Module):
    def __init__(self, in_features, theta=0.0, leakage_factor=0, leakage_period=1, threshold=0.0, reset_to=0.0):
        """
        עטיפה מותאמת אישית של נוירון SCTN כך שיעבוד כרכיב אקטיבציה של PyTorch.
        """
        super(SCTNeuronPyTorch, self).__init__()

        self.theta = theta
        self.leakage_factor = leakage_factor
        self.leakage_period = leakage_period
        self.threshold = threshold
        self.reset_to = reset_to

        # זיכרון פנימי של הנוירון (ה-Membrane Potential והטיימר)
        self.register_buffer('membrane_potential', torch.zeros(1))
        self.register_buffer('leakage_timer', torch.zeros(1))

        # למרות שאנחנו עוטפים SNN, PyTorch מצפה לשכבה ליניארית/קונבולוציה לפני
        # אבל כדי להיות תואמים לממשק, נאפשר הגדרת משקולות פנימית אם צריך
        self.linear = nn.Linear(in_features, 1, bias=False)

    def reset_state(self):
        """חובה לקרוא לפונקציה הזו לפני כל סדרה (Batch) חדשה של זמן"""
        self.membrane_potential.zero_()
        self.leakage_timer.zero_()

    def forward(self, x):
        """
        מחליף את פונקציית ctn_cycle.
        x: זרם כניסה מהשכבה הקודמת (למשל אחרי nn.Conv2d).
        """
        # 1. חישוב זרם הכניסה (Input Current) והוספת Theta
        current = self.linear(x) + self.theta

        # 2. עדכון פוטנציאל הממברנה בהתאם ל-Leakage Factor (כפי שכתוב ב-_kernel)
        if self.leakage_factor < 3:
            self.membrane_potential += current
        else:
            lf_mult = (2 ** (self.leakage_factor - 3))
            self.membrane_potential += current * lf_mult

        # 3. בדיקת אקטיבציה (BINARY - סף ירייה)
        # ב-PyTorch כדי להעביר גרדיאנטים (Surrogate Gradient) כותבים ככה:
        spikes = (self.membrane_potential > self.threshold).float()

        # 4. מנגנון הדליפה (Leakage Period)
        self.leakage_timer += 1
        mask = (self.leakage_timer >= self.leakage_period).float()

        # חישוב הדעיכה (Decay) לפי הנוסחה המדויקת מהקוד המקורי
        decay_delta = torch.where(
            self.membrane_potential < 0,
            (-self.membrane_potential) / (2 ** self.leakage_factor),
            -(self.membrane_potential / (2 ** self.leakage_factor))
        )

        # החלת הדעיכה ואיפוס טיימר איפה שהטיימר עבר את התקופה
        self.membrane_potential = self.membrane_potential + (decay_delta * mask)
        self.leakage_timer = self.leakage_timer * (1 - mask)

        # 5. איפוס הממברנה אחרי ירייה (Reset)
        self.membrane_potential = torch.where(
            spikes > 0,
            torch.full_like(self.membrane_potential, self.reset_to),
            self.membrane_potential
        )

        return spikes