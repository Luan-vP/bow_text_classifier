import os
from logging import Logger
from pathlib import Path

from fastapi import FastAPI, Response
from pydantic import BaseModel

from .nn import BoW_Classifier

logger = Logger(__name__)

MODEL_PACKAGE_DIR = os.getenv("MODEL_PACKAGE_DIR", None)

if MODEL_PACKAGE_DIR is not None:
    MODEL_PACKAGE_DIR = Path(MODEL_PACKAGE_DIR).resolve()
    logger.info("Loading model from")
    if MODEL_PACKAGE_DIR.exists():
        model = BoW_Classifier()
        model.load_model(MODEL_PACKAGE_DIR)
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PACKAGE_DIR}")
else:
    raise FileNotFoundError("MODEL_PACKAGE_DIR environment variable not set")


app = FastAPI()


@app.get("/")
def root():
    return {"Hello": "World!"}


class InferenceInput(BaseModel):
    data: str


@app.post("/inference")
def inference(input: InferenceInput):
    return {"class": model(input.data)}
