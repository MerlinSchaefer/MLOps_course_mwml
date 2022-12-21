# tagifai/main.py
import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import utils
from config import config
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
from tagifai import data, predict, train

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


def train_model(args_filepath, experiment_name, run_name):
    """Train a model given an arguments file"""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_filepath))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID : {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # log metrics and parameters
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), (Path(dp, "args.json")))
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


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


def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


def predict_tag(text, run_id=None):
    """Predict tag for text"""
    if not run_id:
        # use latest run
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    args_filepath = Path(config.CONFIG_DIR, "args.json")
    train_model(args_filepath, experiment_name="test_baselines", run_name="sgd")
    # optimize(args_filepath, study_name="testrun", num_trials=10)
    print("Model trained.")
    text = "Transfer learning with transformers for text classification."
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    predict_tag(text=text, run_id=run_id)
