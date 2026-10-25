from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
raw_dataset_dir = os.getenv("DATASET_DIR", "Capstone_Project_Dataset_FTTT_2026")
DATASET_DIR = Path(raw_dataset_dir).expanduser()
if not DATASET_DIR.is_absolute():
    DATASET_DIR = (ROOT / DATASET_DIR).resolve()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
REFERENCE_DATE = "2026-10-20"
