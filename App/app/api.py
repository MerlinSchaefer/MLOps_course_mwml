from fastapi import FastAPI
from http import HTTPStatus
from typing import Dict


app = FastAPI(
    title="TagifAI - Made with ML",
    description="Classify machine learning projects",
    version="0.1"
)

@app.get("/")
def _index() -> Dict:
    """Health check"""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data":{}
    }