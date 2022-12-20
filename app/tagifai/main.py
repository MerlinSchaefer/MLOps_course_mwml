# tagifai/main.py
import pandas as pd
from pathlib import Path

from config import config
import utils


def elt_data():
    """Extract, load and transform the data assets"""
    # Extract and Load
    projects = pd.read_csv(config.PROJECTS_URL)  # features
    tags = pd.read_csv(config.TAGS_URL)  # labels
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"))
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"))

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df["tag"].notnull()]
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)
    try:
        logger.info("Data extracted and saved.")
    except NameError:
        print("No logger. Continuing without.")
        print("(Fallback logs) Data extracted and saved.")
