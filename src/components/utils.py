import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X, y, X_test, y_test, models, param=None):
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {}) if param else {}

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0, refit=True)
                gs.fit(X, y)
                model = gs.best_estimator_

            model.fit(X, y)
            predictions = model.predict(X_test)
            report[model_name] = r2_score(y_test, predictions)

        return report
    except Exception as e:
        raise CustomException(e, sys)
            