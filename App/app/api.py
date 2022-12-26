from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from config import config
from fastapi import FastAPI, Request
from tagifai import main

app = FastAPI(
    title="TagifAI - Made with ML",
    description="Classify machine learning projects",
    version="0.1",
)


def construct_response(f):
    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        result = f(request, *args, **kwargs)
        response = {
            "message": result["message"],
            "method": request.method,
            "status-code": result["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in result:
            response["data"] = result["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check"""
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    config.logger.info("Ready for inference")


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args/{arg}")
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {arg: vars(artifacts["args"]).get(arg, "")},
    }
    return response

@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response
