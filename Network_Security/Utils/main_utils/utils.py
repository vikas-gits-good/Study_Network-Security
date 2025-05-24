from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train

import os
import yaml
import dill
import pickle
import numpy as np


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


def save_data_as_npy(file_path: str = None, array: np.array = None):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

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
