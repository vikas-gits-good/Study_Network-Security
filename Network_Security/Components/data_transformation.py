from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train
from Network_Security.Entity.config_entity import DataTransformationConfig
from Network_Security.Entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from Network_Security.Utils.main_utils.utils import save_numpy_array, save_object
from Network_Security.Constants.train_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    DATA_TRANSFORMATION_PIPELINE_OBJECT_FILE_NAME,
)

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(
        self,
        data_vald_artf: DataValidationArtifact,
        data_trfm_conf: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_vald_artf = data_vald_artf
            self.data_trfm_conf = data_trfm_conf

        except Exception as e:
            raise NetworkSecurityException(e)

    @staticmethod
    def read_data(file_path: str = None) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e)

    def data_trfm_obj(cls) -> Pipeline:
        try:
            logger_train.info(
                "Data_Transformation: Data preprocessing pipeline creation started"
            )
            ppln_prpc: Pipeline = Pipeline(
                [
                    ("KNNImputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ]
            ).set_output(transform="pandas")
            logger_train.info(
                "Data_Transformation: Data preprocessing pipeline creation finished"
            )
            return ppln_prpc

        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_trfm(self) -> DataTransformationArtifact:
        try:
            logger_train.info("Data_Transformation: Train-Test set reading started")
            df_train = DataTransformation.read_data(
                self.data_vald_artf.valid_train_file_path
            )
            df_test = DataTransformation.read_data(
                self.data_vald_artf.valid_test_file_path
            )
            logger_train.info("Data_Transformation: Train-Test set reading finished")

            logger_train.info(
                "Data_Transformation: X & Y Train-Test set splitting started"
            )
            y_train = df_train[TARGET_COLUMN].replace(-1, 0)
            x_train = df_train.drop(columns=[TARGET_COLUMN])
            y_test = df_test[TARGET_COLUMN].replace(-1, 0)
            x_test = df_test.drop(columns=[TARGET_COLUMN])
            logger_train.info(
                "Data_Transformation: X & Y Train-Test set splitting finished"
            )

            logger_train.info(
                "Data_Transformation: Train-Test data fit_transform started"
            )
            ppln_prpc = self.data_trfm_obj()  # get pipeline object
            x_train_trfm = ppln_prpc.fit_transform(x_train)
            x_test_trfm = ppln_prpc.transform(x_test)

            x_train_trfm_ary = np.c_[x_train_trfm, np.array(y_train)]
            x_test_trfm_ary = np.c_[x_test_trfm, np.array(y_test)]
            logger_train.info(
                "Data_Transformation: Train-Test data fit_transform finished"
            )

            logger_train.info(
                "Data_Transformation: Saving Train-Test transformed array started"
            )
            save_numpy_array(
                file_path=self.data_trfm_conf.trfm_train_file_path,
                array=x_train_trfm_ary,
            )
            save_numpy_array(
                file_path=self.data_trfm_conf.trfm_test_file_path, array=x_test_trfm_ary
            )
            save_object(
                file_path=self.data_trfm_conf.trfm_object_file_path, obj=ppln_prpc
            )

            logger_train.info(
                "Data_Transformation: Saving Train-Test transformed array finished"
            )

            data_trfm_artf = DataTransformationArtifact(
                trfm_object_file_path=self.data_trfm_conf.trfm_object_file_path,
                trfm_train_file_path=self.data_trfm_conf.trfm_train_file_path,
                trfm_test_file_path=self.data_trfm_conf.trfm_test_file_path,
            )
            save_object(file_path="final_models/ppln_prpc.pkl", obj=ppln_prpc)

            return data_trfm_artf

        except Exception as e:
            raise NetworkSecurityException(e)
