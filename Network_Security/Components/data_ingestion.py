from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train
from Network_Security.Entity.config_entity import DataIngestionConfig
from Network_Security.Entity.artifact_entity import DataIngestionArtifact


import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ings_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ings_config = data_ings_config

        except Exception as e:
            raise NetworkSecurityException(e)

    def export_to_dataframe(self):
        try:
            db_name = self.data_ings_config.database_name
            cln_name = self.data_ings_config.collection_name
            self.mongo_client = pymongo.MongoClient(mongo_db_url)
            collection = self.mongo_client[db_name][cln_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns="_id", axis=1)
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e)

    def export_to_feat_store(self, data: pd.DataFrame):
        try:
            feat_store_file_path = self.data_ings_config.feat_store_file_path
            dir_path = os.path.dirname(feat_store_file_path)

            os.makedirs(dir_path, exist_ok=True)
            data.to_csv(feat_store_file_path, index=False, header=True)
            return data

        except Exception as e:
            raise NetworkSecurityException(e)

    def data_train_test_split(self, data: pd.DataFrame):
        try:
            split_ratio = self.data_ings_config.train_test_split_ratio

            logger_train.info("Data_Ingestion: Train-Test Split Started")
            df_train, df_test = train_test_split(
                data, test_size=split_ratio, random_state=89
            )
            logger_train.info("Data_Ingestion: Train-Test Split Finished")

            # Create directory to store train.csv and test.csv
            dir_path = os.path.dirname(self.data_ings_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logger_train.info("Data_Ingestion: Saving Train and test sets started")
            df_train.to_csv(
                self.data_ings_config.train_file_path, index=False, header=True
            )
            df_test.to_csv(
                self.data_ings_config.test_file_path, index=False, header=True
            )
            logger_train.info("Data_Ingestion: Saving Train and test sets finished")

        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_ings(self):
        try:
            df = self.export_to_dataframe()
            df = self.export_to_feat_store(data=df)
            self.data_train_test_split(data=df)

            data_ing_artf = DataIngestionArtifact(
                train_file_path=self.data_ings_config.train_file_path,
                test_file_path=self.data_ings_config.test_file_path,
            )
            return data_ing_artf

        except Exception as e:
            raise NetworkSecurityException(e)
