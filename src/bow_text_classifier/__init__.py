import os
from logging import Logger
from pathlib import Path

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

__all__ = ["model", BoW_Classifier]
