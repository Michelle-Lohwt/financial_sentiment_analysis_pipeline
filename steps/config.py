from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
  """Model Configs"""
  ml_model_name: str = "RandomForest"