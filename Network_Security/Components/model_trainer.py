from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train
from Network_Security.Entity.config_entity import ModelTrainerConfig
from Network_Security.Entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from Network_Security.Utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array,
    evaluate_models,
)
from Network_Security.Utils.ml_utils.model.estimator import NetworkModel
from Network_Security.Utils.ml_utils.metric.clasf_metric import get_clasf_score
from Network_Security.Constants.model_params import models_dict, params_dict

import os
import numpy as np
import mlflow
import dagshub


class ModelTrainer:
    def __init__(
        self,
        data_trfm_artf: DataTransformationArtifact = None,
        model_train_conf: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.model_train_conf = model_train_conf
            self.data_trfm_artf = data_trfm_artf
            dagshub.init(
                repo_owner="vikas-gits-good",
                repo_name="Study_Network-Security",
                mlflow=True,
            )

        except Exception as e:
            raise NetworkSecurityException(e)

    def track_mlflow(self, model, metric):
        try:
            with mlflow.start_run():
                metrics = {
                    "f1_score": metric.score_f1,
                    "precision_score": metric.score_precision,
                    "recall_score": metric.score_recall,
                }
                # Track model and its metrics
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise NetworkSecurityException(e)

    def train_models(
        self,
        x_train: np.array = None,
        y_train: np.array = None,
        x_test: np.array = None,
        y_test: np.array = None,
    ):
        try:
            model_report = evaluate_models(
                x_train, y_train, x_test, y_test, models_dict, params_dict
            )
            best_model_name = list(model_report.keys())[0]
            best_model_object = model_report[best_model_name]["best_fit_model"]

            y_train_pred = best_model_object.predict(x_train)
            y_test_pred = best_model_object.predict(x_test)

            logger_train.info("Model_Trainer: Calculation of metrics started")
            clasf_train_metric = get_clasf_score(y_train, y_train_pred)
            clasf_test_metric = get_clasf_score(y_test, y_test_pred)
            logger_train.info("Model_Trainer: Calculation of metrics finished")

            # mlflow tracking
            self.track_mlflow(best_model_object, clasf_train_metric)
            self.track_mlflow(best_model_object, clasf_test_metric)

            logger_train.info("Model_Trainer: Creation of pred pipeline object started")
            ppln_prpc = load_object(self.data_trfm_artf.trfm_object_file_path)
            model_dir_path = self.model_train_conf.best_modl_file_path
            os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
            net_mdl = NetworkModel(pipeline=ppln_prpc, model=best_model_object)
            save_object(file_path=model_dir_path, obj=net_mdl)
            logger_train.info(
                "Model_Trainer: Creation of pred pipeline object finished"
            )
            save_object(file_path="final_models/model.pkl", obj=best_model_object)

            model_train_artf = ModelTrainerArtifact(
                trnd_modl_file_path=self.model_train_conf.best_modl_file_path,
                train_metric_artf=clasf_train_metric,
                test_metric_artf=clasf_test_metric,
            )

            return model_train_artf

        except Exception as e:
            raise NetworkSecurityException(e)

    def init_model_train(self) -> ModelTrainerArtifact:
        try:
            logger_train.info("Model_Trainer: Loading Train-Test array started")
            train_file_path = self.data_trfm_artf.trfm_train_file_path
            test_file_path = self.data_trfm_artf.trfm_test_file_path
            ary_train = load_numpy_array(train_file_path)
            ary_test = load_numpy_array(test_file_path)
            logger_train.info("Model_Trainer: Loading Train-Test array finished")

            x_train, y_train, x_test, y_test = (
                ary_train[:, :-1],
                ary_train[:, -1],
                ary_test[:, :-1],
                ary_test[:, -1],
            )
            logger_train.info("Model_Trainer: Full model training started")
            model_train_artf = self.train_models(x_train, y_train, x_test, y_test)
            logger_train.info("Model_Trainer: Full model training finished")

            return model_train_artf

        except Exception as e:
            raise NetworkSecurityException(e)
