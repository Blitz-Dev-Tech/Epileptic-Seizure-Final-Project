import os
import mne
import matplotlib.pyplot as plt

# ייבוא הגדרות מהקונפיגורציה שלכם
import config


def plot_raw_eeg(file_name, start_time_sec=0, duration_sec=10, channels=23):
    """
    טוען קובץ EDF ומציג את אותות ה-EEG הגולמיים.

    :param file_name: שם קובץ ה-EDF (למשל 'chb01_03.edf'). יש להניח שהוא תחת config.RAW_DIR.
    :param start_time_sec: זמן ההתחלה להצגה בשניות.
    :param duration_sec: משך הזמן להצגה בשניות.
    :param channels: מספר הערוצים להצגה (ברירת מחדל 23).
    """
    # חיפוש הקובץ בתיקיית ה-RAW
    file_path = None
    for root, _, files in os.walk(config.RAW_DIR):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            break

    if not file_path:
        print(f"[!] Error: File '{file_name}' not found in {config.RAW_DIR} or its subdirectories.")
        return

    print(f"[*] Loading raw EEG data from: {file_name}")

    # טעינת הקובץ בעזרת MNE (ללא פילטור, מציג את האות כפי שהוא)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # חיתוך מספר הערוצים למה שביקשנו (בדרך כלל 23)
    if raw.info['nchan'] > channels:
        raw.pick_channels(raw.ch_names[:channels])

    print(f"[*] Displaying {channels} channels from {start_time_sec}s to {start_time_sec + duration_sec}s")

    # קביעת צבעוניות אחידה לקריאה ברורה יותר
    color_dict = {ch: 'blue' for ch in raw.ch_names}

    # יצירת חלון הגרף בעזרת פונקציית plot המובנית של MNE
    # היא תומכת בגלילה וזום אינטראקטיבי
    fig = raw.plot(
        start=start_time_sec,
        duration=duration_sec,
        n_channels=channels,
        color=color_dict,
        title=f"Raw EEG Data - {file_name}",
        show=False,  # לא מציג מיד כדי שנוכל לשמור
        scalings='auto'  # התאמת קנה מידה אוטומטית לעוצמת האותות
    )

    # וידוא שתיקיית plots קיימת
    plots_dir = os.path.join(config.BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # שמירת הגרף כתמונה למצגת
    save_path = os.path.join(plots_dir, f"Raw_EEG_{file_name.replace('.edf', '')}.png")
    fig.savefig(save_path, dpi=300)
    print(f"[*] Plot saved to: {save_path}")

    # הצגת הגרף על המסך (מאפשר אינטראקציה אם מריצים לוקאלית)
    plt.show()


if __name__ == "__main__":
    # הדגמה: הציגו את 10 השניות הראשונות של מטופל שידוע שיש לו התקף
    # החליפו בשם קובץ שקיים אצלכם בתיקיית ה-raw (למשל מהרשימה שסימננו בטבלה קודם)
    sample_file = "chb01_26.edf"

    # אפשר לשנות את ה-start_time כדי "לטייל" לאורך הרישום
    plot_raw_eeg(sample_file, start_time_sec=0, duration_sec=10)