import os

# נתיב בסיס של הפרויקט
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# נתיבי נתונים
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# הגדרות עיבוד אותות
WINDOW_SIZE_SEC = 4
OVERLAP_SEC = 2  # חפיפה של 50% (2 שניות)
FS = 256         # תדר דגימה

# הגדרות אימון
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 15
DEVICE = "cuda" # ישונה אוטומטית ל-cpu בתוך הקוד אם אין כרטיס מסך