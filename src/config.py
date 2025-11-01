from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_PATH = DATA_DIR / "diabetic_data.csv"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train.csv"
VAL_SPLIT_PATH = PROCESSED_DIR / "val.csv"
TEST_SPLIT_PATH = PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "readmission_rf.joblib"

# Columns with excessive missing values to drop
DROP_COLS = [
    "weight",
    "payer_code",
    "medical_specialty",
]

# Target and key columns
TARGET_COL = "readmitted"
ID_COLS = ["encounter_id", "patient_nbr"]
LENGTH_OF_STAY_COL = "time_in_hospital"

# Medication columns in the dataset (intersected with actual columns at runtime)
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
