import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Root project: MovieSearch/
ROOT_DIR = Path(__file__).resolve().parents[1]

# ==========================
# ChromaDB config
# ==========================
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# ==========================
# Data paths
# ==========================
DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

MOVIE_FOLDERS = RAW_DATA_DIR / "DataMovie"
RAW_JSON_PATH = RAW_DATA_DIR / "movies_data.json"
CLEAN_EN_JSON_PATH = PROCESSED_DATA_DIR / "movies_data_english_clean.json"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"

# ==========================
# Results paths
# ==========================
RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = RESULTS_DIR / "logs"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

# ==========================
# Convert Path -> str
# để tương thích với code cũ đang dùng os.path / open()
# ==========================
ROOT_DIR = str(ROOT_DIR)

DATA_DIR = str(DATA_DIR)
RAW_DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)
INDEX_DIR = str(INDEX_DIR)

MOVIE_FOLDERS = str(MOVIE_FOLDERS)
RAW_JSON_PATH = str(RAW_JSON_PATH)
CLEAN_EN_JSON_PATH = str(CLEAN_EN_JSON_PATH)
BM25_PATH = str(BM25_PATH)

RESULTS_DIR = str(RESULTS_DIR)
LOG_DIR = str(LOG_DIR)