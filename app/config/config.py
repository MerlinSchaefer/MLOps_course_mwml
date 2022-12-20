# config.py
from pathlib import Path

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
TAGS_URL = (
    "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
)


# Data preprocessing
ACCEPTED_TAGS = [
    "natural-language-processing",
    "computer-vision",
    "mlops",
    "graph-learning",
]

nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
stemmer = PorterStemmer()
