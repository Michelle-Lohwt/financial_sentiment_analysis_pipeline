import logging
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
  Annotated[spmatrix, "X_train"],
  Annotated[spmatrix, "X_test"],
  Annotated[np.ndarray, "y_train"],
  Annotated[np.ndarray, "y_test"],
]:
  """
  Cleans the data and divides it into train and test
  
  Args:
    df: Raw data
  Returns:
    X_train: Training data
    X_test: Testing data
    y_train: Training labels
    y_test: Testing labels
  """
  try:
    process_strategy = DataPreProcessStrategy()
    data_cleaning = DataCleaning(df, process_strategy)
    processed_data = data_cleaning.handle_data()
    
    divide_strategy = DataDivideStrategy()
    data_cleaning = DataCleaning(processed_data, divide_strategy)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    logging.info("Data cleaning completed")
    
    return X_train, X_test, y_train, y_test
  
  except Exception as e:
    logging.error("Error in cleaning data: {}".format(e))
    raise e