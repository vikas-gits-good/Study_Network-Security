import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
from pymongo.mongo_client import MongoClient
import numpy as np
import pymongo

from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train

load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
# only allow certified connection requests. Certificate Authority
ca = certifi.where()


class NetworkData_Extract:
    def __init__(self, file_path, database, collection):
        try:
            self.file_path = file_path
            self.database = database
            self.collection = collection

        except Exception as e:
            raise NetworkSecurityException(e)

    def csv_to_json(self) -> json:
        try:
            logger_train.info("ETL_Extract Started")
            data = pd.read_csv(self.file_path)
            data.reset_index(drop=True, inplace=True)
            logger_train.info("ETL_Extract Finished")

            # Convert csv data to a list of json. Each row is now a dictionary tht is appended to a list
            logger_train.info("ETL_Transform Started")
            records = list(json.loads(data.T.to_json()).values())
            logger_train.info("ETL_Transform Finished")

            return records

        except Exception as e:
            raise NetworkSecurityException(e)

    def insert_data_to_db(self, data):
        try:
            logger_train.info("ETL_Load Started")
            self.data = data
            self.mongo_client = MongoClient(mongo_db_url)

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.data)
            logger_train.info("ETL_Load Finished")

            return len(self.data)

        except Exception as e:
            raise NetworkSecurityException(e)


if __name__ == "__main__":
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "Study_NetworkSecurity_DB"
    Collection = "NetworkData"

    nwk_obj = NetworkData_Extract(
        file_path=FILE_PATH, database=DATABASE, collection=Collection
    )
    data = nwk_obj.csv_to_json()
    nwk_obj.insert_data_to_db(data)
