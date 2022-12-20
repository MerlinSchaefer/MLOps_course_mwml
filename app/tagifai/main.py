# tagifai/main.py
import json
import warnings
from argparse import Namespace
from pathlib import Path

import pandas as pd
import utils
from config import config
from tagifai import data, train

warnings.filterwarnings("ignore")  # necessary for SGD max_iter warning


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


def train_model(args_filepath):
    """Train a model given an arguments file"""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_filepath))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))
    print(df.head())


if __name__ == "__main__":
    args_filepath = Path(config.CONFIG_DIR, "args.json")
    train_model(args_filepath)
