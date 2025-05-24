from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train
from Network_Security.Entity.config_entity import DataValidationConfig
from Network_Security.Entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from Network_Security.Constants.train_pipeline import SCHEMA_FILE_PATH
from Network_Security.Utils.main_utils.utils import read_yaml_file, write_yaml_file

import os
import pandas as pd
from typing import List, Tuple
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(
        self,
        data_ing_artf: DataIngestionArtifact,
        data_vald_conf: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ing_artf = data_ing_artf
            self.data_vald_conf = data_vald_conf
            self.schema_conf = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e)

    @staticmethod
    def read_data(file_path: str = None) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e)

    def validate_columns(self, data: List[pd.DataFrame] = None) -> Tuple[bool]:
        try:
            req_nu_of_cols = len(self.schema_conf["columns"])
            exs_nu_cols = [len(df.columns) for df in data]
            cols_dtype = tuple(
                True if df.dtypes.unique() != "O" else False for df in data
            )

            logger_train.info(
                f"Data_Validation: Required number of columns = {req_nu_of_cols}"
            )
            logger_train.info(
                f"Data_Validation: Training number of columns = {exs_nu_cols[0]}, Test number of columns = {exs_nu_cols[1]}"
            )

            return tuple(x == req_nu_of_cols for x in exs_nu_cols) + cols_dtype

        except Exception as e:
            raise NetworkSecurityException(e)

    def detect_data_drift(
        self,
        df_base: pd.DataFrame = None,
        df_crnt: pd.DataFrame = None,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report = {}

            for col in df_base.columns:
                d1 = df_base[col]
                d2 = df_crnt[col]
                similarity = ks_2samp(d1, d2)
                if threshold <= similarity.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update(
                    {
                        col: {
                            "p_value": float(similarity.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            # Create data drift report
            drift_report_file_path = self.data_vald_conf.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_vald(self) -> DataIngestionArtifact:
        try:
            train_file_path = self.data_ing_artf.train_file_path
            test_file_path = self.data_ing_artf.test_file_path

            # Read data
            logger_train.info("Data_Validation: Reading of Train and Test set started")
            df_train = DataValidation.read_data(train_file_path)
            df_test = DataValidation.read_data(test_file_path)
            logger_train.info("Data_Validation: Reading of Train and Test set finished")

            # validate data
            logger_train.info("Data_Validation: Validation of train-test set started")
            stat_train, stat_test, num_train, num_test = self.validate_columns(
                data=[df_train, df_test]
            )
            logger_train.info("Data_Validation: Validation of train-test set finished")
            logger_train.info(
                f"Data_Validation: Validation[Train] = {stat_train}, Validation[Test] = {stat_test}"
            )
            logger_train.info(
                f"Data_Validation: Numeric[Train] = {num_train}, Numeric[Test] = {num_test}"
            )

            # if all data are valid, then check data drft and if thats okay then proceed forward
            # if data invalid or data drift fails, send data to invalid data and return None
            fail = False
            if fail:
                data_vald_artf = DataValidationArtifact(
                    validation_status=None,
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=None,
                )
                return data_vald_artf

            # check data drift
            status = self.detect_data_drift(df_base=df_train, df_crnt=df_test)
            dir_path = os.path.dirname(self.data_vald_conf.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save validated data to file
            # if validation passes
            df_train.to_csv(
                self.data_vald_conf.valid_train_file_path, index=False, header=True
            )
            df_test.to_csv(
                self.data_vald_conf.valid_test_file_path, index=False, header=True
            )
            # if validation fails

            # Export artifact
            data_vald_artf = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ing_artf.train_file_path,
                valid_test_file_path=self.data_ing_artf.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_vald_conf.drift_report_file_path,
            )

            return data_vald_artf

        except Exception as e:
            raise NetworkSecurityException(e)
