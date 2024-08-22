import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class Model(ABC):
  """
  Abstract class for all models
  """
  @abstractmethod
  def train(self, X_train, y_train):
    """
    Trains the model
    
    Args:
      X_train: Training data
      y_train: Training labels
    Returns:
      None
    """
    pass
  
class RandomForestModel(Model):
  """
  Random Forest model
  """
  def train(self, X_train, y_train, **kwargs):
    """
    Trains the model
    
    Args:
      X_train: Training data
      y_train: Training labels
    Returns:
      None
    """
    try:
      rf = RandomForestClassifier(**kwargs)
      rf.fit(X_train, y_train)
      logging.info("Model training completed")
      return rf
    except Exception as e:
      logging.error("Error in training model: {}".format(e))
      raise e