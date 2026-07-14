FATIGUE_MODEL_PATH = "models/fatigue_best.pt"   
UNSAFE_MODEL_PATH = "models/unsafe_best.pt"    

CONF_THRESHOLD_FATIGUE = 0.5
CONF_THRESHOLD_UNSAFE = 0.4

FATIGUE_CLASS_NAMES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']

MOUTH_OPEN_DURATION = 2.0     # detik = menguap
MOUTH_OPEN_EVENT_LIMIT = 3    # jumlah menguap sebelum status "Pengendara Lelah"

EYE_CLOSED_DURATION = 1.0     # detik = mengantuk
EYE_CLOSED_EVENT_LIMIT = 2    # jumlah mengantuk sebelum status "Pengendara Mengantuk"

MISS_TOLERANCE = 5            # toleransi "kedip" supaya durasi tidak reset
RESET_WINDOW = 60.0           # detik idle sebelum counter di-reset otomatis
ALERT_DISPLAY_DURATION = 5.0  # lama alert tampil di layar sebelum hilang & reset

FATIGUE_BOX_COLORS = {
    'closed_eye':   (0, 0, 255),      # merah
    'closed_mouth': (0, 255, 0),      # hijau
    'open_eye':     (255, 200, 0),    # biru muda
    'open_mouth':   (0, 165, 255),    # oranye
}

AUDIO_ALERT_1 = "audio/audio1.mp3"   
AUDIO_ALERT_2 = "audio/audio2.mp3"

UNSAFE_ALERT_CLASSES = ['smoking', 'vaping', 'Phone']
UNSAFE_ALERT_COOLDOWN = 3.0     

UNSAFE_BOX_COLORS = {
    'smoking': (0, 128, 255),   # oranye
    'vaping':  (255, 0, 255),   # magenta
    'Phone':   (0, 0, 255),     # merah
}
UNSAFE_BOX_COLOR_DEFAULT = (0, 255, 255)  

UNSAFE_ALERT_SOUND_PATH = None
UNSAFE_ALERT_BEEP_FREQ_HZ = 1200
UNSAFE_ALERT_BEEP_DURATION_MS = 350

OUTPUT_DIR = "output"