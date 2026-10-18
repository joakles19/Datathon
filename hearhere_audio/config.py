"""Shared audio and MFCC settings used by training and inference."""

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

N_MFCC = 20
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 128
RESAMPLE_TYPE = "soxr_hq"

CLASSES = (
    "car_horn",
    "clock_alarm",
    "crying_baby",
    "door_wood_knock",
    "engine",
    "glass_breaking",
    "rain",
    "siren",
    "vacuum_cleaner",
    "wind",
)
# v2 expands the classifier vocabulary while excluding keyboard typing.
MODEL_VERSION = "hearhere-rf-v2"
ALERT_CLASSES = (
    "door_wood_knock",
    "siren",
    "glass_breaking",
    "clock_alarm",
    "crying_baby",
    "car_horn",
)

FEATURE_ORDER = tuple(
    [f"mfcc_{index}_mean" for index in range(N_MFCC)]
    + [f"mfcc_{index}_std" for index in range(N_MFCC)]
)
