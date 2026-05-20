"""Project-wide paths and hyperparameters."""

from pathlib import Path

# Reproducibility
RANDOM_STATE = 42

# Train / test split
TEST_SIZE = 0.2

# Collaborative filtering
K_NEIGHBORS = 20
MIN_COMMON_RATINGS = 5
SIMILARITY_METRIC = "cosine"  # "cosine" or "pearson"

# Matrix factorization (sklearn TruncatedSVD)
SVD_N_COMPONENTS = 50
SVD_N_ITER = 5

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "ml-100k"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RATINGS_FILE = DATA_RAW / "u.data"
MOVIES_FILE = DATA_RAW / "u.item"
GENRE_FILE = DATA_RAW / "u.genre"

RATING_SCALE = (1, 5)
