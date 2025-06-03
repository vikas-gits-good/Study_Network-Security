from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train

import os
import yaml
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def read_yaml_file(file_path: str = None) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e)


def write_yaml_file(
    file_path: str = None, content: object = None, replace: bool = False
) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)

    except Exception as e:
        raise NetworkSecurityException(e)


def save_numpy_array(file_path: str = None, array: np.array = None):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e)


def load_numpy_array(file_path: str = None) -> np.array:
    try:
        logger_train.info("Utils: load numpy array started")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        logger_train.info("Utils: load numpy array finished")

    except Exception as e:
        raise NetworkSecurityException(e)


def save_object(file_path: str = None, obj: object = None):
    try:
        logger_train.info("Utils: save object started")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger_train.info("Utils: save object finished")

    except Exception as e:
        raise NetworkSecurityException(e)


def load_object(file_path: str = None) -> object:
    try:
        logger_train.info("Utils: load object started")
        if not os.path.exists(file_path):
            raise Exception(f"The given path [{file_path}] does not exist")
        with open(file_path, "rb") as file_obj:
            item = pickle.load(file_obj)
            logger_train.info("Utils: load object finished")
            return item

    except Exception as e:
        raise NetworkSecurityException(e)


def evaluate_models(
    x_train, y_train, x_test, y_test, models: dict, params: dict
) -> dict:
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param_dict = params[model_name]

            logger_train.info(f"Model_Trainer: Training of '{model_name}' started")
            gs = GridSearchCV(model, param_dict, cv=3, refit=True)
            gs.fit(x_train, y_train)
            logger_train.info(f"Model_Trainer: Training of '{model_name}' finished")

            y_train_pred = gs.predict(x_train)
            y_test_pred = gs.predict(x_test)

            logger_train.info(f"Model_Trainer: Scoring of '{model_name}' started")
            train_score = f1_score(y_train, y_train_pred)
            test_score = f1_score(y_test, y_test_pred)
            logger_train.info(f"Model_Trainer: Scoring of '{model_name}' finished")

            report[model_name] = {
                "best_fit_model": gs,
                "f1_score_train": train_score,
                "f1_score_test": test_score,
            }
        report_sorted = dict(
            sorted(
                report.items(), key=lambda item: item[1]["f1_score_test"], reverse=True
            )
        )
        return report_sorted

    except Exception as e:
        raise NetworkSecurityException(e)
