import pandas as pd
import os
import numpy as np
from mlProject import logger
from sklearn.linear_model import LogisticRegression
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        train_y = np.ravel(train_y)
        test_y = test_data[[self.config.target_column]]

        lr = LogisticRegression(C=self.config.C, class_weight=self.config.class_weight, l1_ratio=self.config.l1_ratio,max_iter=self.config.max_iter, penalty=self.config.penalty, solver=self.config.solver,tol=self.config.tol)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

