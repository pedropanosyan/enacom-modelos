import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class Setup:
    def __init__(self) -> None:
        pass

    def get_train_data_unsupervised(
        self, 
        clean_data: np.ndarray, 
        anomaly_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Split the clean data into a training set and a test set
        clean_train, clean_test = train_test_split(clean_data, test_size=0.2)
        
        # Split the anomaly data into a training set and a test set
        _, anomaly_test = train_test_split(anomaly_data, test_size=0.2)
        
        # The training set for the model should only be the clean data
        data_train = clean_train
        
        # The test set for evaluation should be a combination of clean and anomaly data
        data_test = np.concatenate((clean_test, anomaly_test), axis=0)

        # To evaluate, you need to know which is which, so return the separated test sets
        return data_train, clean_test, anomaly_test, data_test

    def get_train_data_supervised(
        self, 
        clean_data: np.ndarray, 
        anomaly_data: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
      clean_df = pd.DataFrame(clean_data)
      anomaly_df = pd.DataFrame(anomaly_data)
      
      clean_df['label'] = 0
      anomaly_df['label'] = 1
      
      all_data = pd.concat([clean_df, anomaly_df], ignore_index=True)
      
      X = all_data.drop('label', axis=1)
      Y = all_data['label']
      
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
      
      return X_train, X_test, Y_train, Y_test    
    

    def get_test_data(self) -> None:
        pass
