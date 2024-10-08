import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, accuracy_score, precision_score

class Evaluation(ABC):
  """
  Abstract class defining strategy for evaluation our models.
  """
  @abstractmethod
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the scores for the model

    Args:
      y_true (np.ndarray): True labels
      y_pred: Predicted labels
    Returns:
      None
    """
    pass
  
class Accuracy(Evaluation):
  """
  Evaluation Strategy that uses Accuracy
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating Accuracy")
      accuracy = accuracy_score(y_true, y_pred)
      logging.info("Accuracy: {}".format(accuracy))
      return accuracy
    except Exception as e:
      logging.error("Error in calculating Accuracy: {}".format(e))
      raise e
    
class Precision(Evaluation):
  """
  Evaluation Strategy that uses Precision
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating Precision")
      precision = precision_score(y_true, y_pred, average='weighted')
      logging.info("Precision: {}".format(precision))
      return precision
    except Exception as e:
      logging.error("Error in calculating Precision: {}".format(e))
      raise e
  
class MSE(Evaluation):
  """
  Evaluation Strategy that uses Mean Squared Error
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating MSE")
      mse = mean_squared_error(y_true, y_pred)
      logging.info("MSE: {}".format(mse))
      return mse
    except Exception as e:
      logging.error("Error in calculating MSE: {}".format(e))
      raise e
    
class R2(Evaluation):
  """
  Evaluation Strategy that uses R2 Score
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating R2 Score")
      r2 = r2_score(y_true, y_pred)
      logging.info("R2 Score: {}".format(r2))
      return r2
    except Exception as e:
      logging.error("Error in calculating R2 Score: {}".format(e))
      raise e
    
class RMSE(Evaluation):
  """
  Evaluation Strategy that uses Root Mean Square Error
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating RMSE")
      rmse = root_mean_squared_error(y_true, y_pred)
      logging.info("R2 Score: {}".format(rmse))
      return rmse
    except Exception as e:
      logging.error("Error in calculating R2 Score: {}".format(e))
      raise e