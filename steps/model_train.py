import logging
import mlflow
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from zenml import step
from src.model_dev import RandomForestModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
  X_train: spmatrix, 
  X_test: spmatrix, 
  y_train: np.ndarray, 
  y_test: np.ndarray,
  config: ModelNameConfig
  ) -> ClassifierMixin:
  """
  Trains the model on the ingested data.
  
  Args:
    X_train: spmatrix, 
    X_test: spmatrix, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
  """
  try:
    model = None
    if config.ml_model_name == "RandomForest":
      mlflow.sklearn.autolog()
      model = RandomForestModel()
      trained_model = model.train(X_train, y_train)
      return trained_model
    else:
      raise ValueError("Model {} not supported".format(config.ml_model_name))
  except Exception as e:
    logging.error("Error in training model: {}".format(e))
    raise e