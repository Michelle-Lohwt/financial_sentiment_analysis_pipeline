import logging
import mlflow
import pandas as pd
import numpy as np
from zenml import step
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from src.evaluation import Accuracy, Precision, MSE, R2, RMSE
from typing import Tuple
from typing_extensions import Annotated

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
  model: ClassifierMixin,
  X_test: spmatrix,
  y_test: np.ndarray
  ) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    # Annotated[float, "r2_score"],
    # Annotated[float, "rmse"]
  ]:
  """
  Evaluates the model on the ingested Data
  
  Args:
    df: the ingested data
  """
  try:
    prediction = model.predict(X_test)
    
    accuracy_class = Accuracy()
    accuracy = accuracy_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("accuracy", accuracy)
    
    precision_class = Precision()
    precision = precision_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("accuracy", precision)
    
    # mse_class = MSE()
    # mse = mse_class.calculate_scores(y_test, prediction)
    # mlflow.log_metric("mse", mse)
    
    # r2_class = R2()
    # r2_score = r2_class.calculate_scores(y_test, prediction)
    # mlflow.log_metric("r2", r2_score)
    
    # rmse_class = RMSE()
    # rmse = rmse_class.calculate_scores(y_test, prediction)
    # mlflow.log_metric("rmse", rmse)
    
    return accuracy, precision
  except Exception as e:
    logging.error("Error in evaluating model: {}".format(e))
    raise e