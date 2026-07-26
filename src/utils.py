import os
import sys
import dill

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)   # retrurns name of the folder

        os.makedirs(dir_path, exist_ok=True)       # creates a folder if it doesn't already exists

        with open(file_path, "wb") as file_obj:      # open in write binary , dill saves binary data
            dill.dump(obj,file_obj)                 # It writes the Python object into the file.

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report= {}
        best_params = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            best_params[model_name] = gs.best_params_

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)   # Train model

            y_train_pred=model.predict(X_train)

            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report, best_params


    except Exception as e:
        raise CustomException(e,sys)    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys) 

