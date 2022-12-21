# tagifai/main.py
import json
import warnings
from argparse import Namespace
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import utils
from config import config
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
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


def optimize(
    args_filepath,
    num_trials,
    study_name="optimization",
):
    """Optimize hyperparameters"""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))
    # Load args
    args = Namespace(**utils.load_dict(filepath=args_filepath))
    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name="optimization", direction="maximize", pruner=pruner
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1"
    )
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict(
        {**args.__dict__, **study.best_trial.params}, args_filepath, cls=NumpyEncoder
    )


if __name__ == "__main__":
    args_filepath = Path(config.CONFIG_DIR, "args.json")
    # train_model(args_filepath)
    optimize(args_filepath, study_name="testrun", num_trials=10)
