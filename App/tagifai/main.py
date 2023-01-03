# tagifai/main.py
import json
import sys
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

sys.path.append("..")
import joblib
import mlflow
import optuna
import pandas as pd
import typer
from config import config
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
from tagifai import data, predict, train, utils

warnings.filterwarnings("ignore")  # necessary for SGD max_iter warning

app = typer.Typer()


@app.command()
def elt_data():
    """Extract, load and transform the data assets"""
    # Extract and Load
    projects = pd.read_csv(config.PROJECTS_URL)  # features
    tags = pd.read_csv(config.TAGS_URL)  # labels
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df["tag"].notnull()]
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)
    try:
        config.logger.info("Data extracted and saved.")
    except NameError:
        print("No logger. Continuing without.")
        print("(Fallback logs) Data extracted and saved.")


@app.command()
def train_model(
    args_filepath: str = "../config/args.json",
    experiment_name: str = "testrun",
    run_name: str = "test",
    test_run: bool = False,
) -> None:
    """Train a model given arguments. Log metrics and save artifacts in model registry.
    Args:
        args_filepath (str): location of args JSON.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_filepath))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        try:
            config.logger.info("Model Training")
            config.logger.info(f"Run ID : {run_id}")
            config.logger.info(json.dumps(performance, indent=2))
        except NameError:
            print("No logger. Continuing without.")
            print(f"Run ID : {run_id}")
            print(json.dumps(performance, indent=2))

        # log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(
                vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder
            )
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
    args_filepath: str = "config/args.json",
    num_trials: int = 20,
    study_name: str = "optimization",
) -> None:
    """Optimize hyperparameters. Log metrics and save best run in logs and args JSON.
    Args:
        args_filepath (str): location of args JSON.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))
    # Load args
    args = Namespace(**utils.load_dict(filepath=args_filepath))
    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
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
    best_trial_dict = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(best_trial_dict, args_filepath, cls=NumpyEncoder)
    config.logger.info(f"Optimization concluded with best trial:\n {best_trial_dict}")


def load_artifacts(run_id: str) -> Dict:
    """Load artifacts for a given run_id.
    Args:
        run_id (str): id of run to load artifacts from.
    Returns:
        Dict: run's artifacts.
    """
    # Locate specifics artifacts directory
    try:
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    except mlflow.exceptions.MlflowException:
        print("Could not find run. Falling back to latest run.")
        experiments = mlflow.list_experiments()
        all_runs = mlflow.search_runs(
            experiment_ids=experiments[-1].experiment_id, order_by=["metric.f1"]
        )
        run_id = all_runs.iloc[-1].run_id
        print(run_id)
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


@app.command()
def predict_tag(text: str = "", run_id: str = None) -> List[Dict[str, str]]:
    """Predict tag for text.

    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.

    Returns:
        List[Dict[str,str]]: Predictions in the form of:
        ```python
        [{
            "input_text": text,
            "predicted_tags": tag,
        },
        ...]
    ```
    """
    if not run_id:
        # use latest run
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()
#    elt_data()
#    args_filepath = Path(config.CONFIG_DIR, "args.json")
#    # train_model(args_filepath, experiment_name="test_baselines", run_name="sgd")
#    optimize(args_filepath, study_name="testrun", num_trials=10)
#    print("Model trained.")
#    text = "Transfer learning with transformers for text classification."
#    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
#    predict_tag(text=text, run_id=run_id)
