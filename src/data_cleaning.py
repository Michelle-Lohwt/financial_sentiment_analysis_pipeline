import logging
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from typing import Union, Tuple

import numpy as np
import pandas as pd
import emoji
import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from wordcloud import STOPWORDS

from nltk import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
  """
  Abstract class defining strategy for handling data
  """
  @abstractmethod
  def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Tuple[spmatrix, spmatrix, np.ndarray, np.ndarray]]:
    pass
  
class DataPreProcessStrategy(DataStrategy):
  """
  Strategy for preprocessing data
  """
  def clean_text(self, df, field):
    df[field] = df[field].str.replace(r"http\S+"," ")
    df[field] = df[field].str.replace(r"http"," ")
    df[field] = df[field].str.replace(r"@","at")
    df[field] = df[field].str.replace("#[A-Za-z0-9_]+", ' ')
    df[field] = df[field].str.replace(r"[^A-Za-z(),!?@\'\"_\n]"," ")
    df[field] = df[field].str.lower()
    
    return df
  
  def preprocess_text(self, text):
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm','im', 'll', 'y', 've', 
                          'u', 'ur', 'don','p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                          'de', 're', 'amp', 'will'])
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('[^a-zA-Z]',' ',text)
    
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"

        # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r'[^\x00-\x7f]','',text)
    text = " ".join([stemmer.stem(word) for word in text.split()])
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(STOPWORDS)]
    text = ' '.join(text)
    return text
  
  
  def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data
    """
    try:
      labelencoder = LabelEncoder()
      data["label_enc"] = labelencoder.fit_transform(data["Sentiment"])
      data.rename(columns={"label": "label_desc"}, inplace=True)
      data.rename(columns={"label_enc": "labels"}, inplace=True)
      data.drop_duplicates(subset=['Sentence'],keep='first',inplace=True)
      
      cleaned_data = self.clean_text(data, "Sentence")
      cleaned_data["Sentence"] = cleaned_data["Sentence"].apply(self.preprocess_text)
      return cleaned_data
    except Exception as e:
      logging.error("Error in preprocessing data: {}".format(e))
      raise e
    
class DataDivideStrategy(DataStrategy):
  """
  Strategy for preprocessing data
  """
  def handle_data(self, data: pd.DataFrame) -> Tuple[spmatrix, spmatrix, np.ndarray, np.ndarray]:
    """
    Divide data into train and test
    """
    try:
      # X_train, X_test, y_train, y_test = train_test_split(data["Sentence"],data["labels"], test_size=0.25, random_state=42)
      X_train, X_test, y_train, y_test = train_test_split(np.array(data["Sentence"]),np.array(data["labels"]), test_size=0.25, random_state=42)
      
      tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize,min_df=0.00002,max_df=0.70)
      X_train_tf = tfidf.fit_transform(X_train.astype('U'))
      X_test_tf = tfidf.transform(X_test.astype('U'))
      
      return X_train_tf, X_test_tf, y_train, y_test
    
    except Exception as e:
      logging.error("Error in dividing data: {}".format(e))
      raise e
    
class DataCleaning:
  """
  Class for cleaning data which processes the data and divides it into train and test
  """
  def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
    self.data = data
    self.strategy = strategy
    
  def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[spmatrix, spmatrix, np.ndarray, np.ndarray]]:
    """
    Handle data
    """
    try:
      return self.strategy.handle_data(self.data)
    except Exception as e:
      logging.error("Error in handling data: {}".format(e))
      raise e