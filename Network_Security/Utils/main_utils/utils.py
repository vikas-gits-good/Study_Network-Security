from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train

import os
import yaml
import dill
import pickle


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
