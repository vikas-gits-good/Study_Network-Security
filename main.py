from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train
from Network_Security.Components.data_ingestion import DataIngestion
# from Network_Security.Entity.config_entity import (
#     DataIngestionConfig,
#     TrainingPipelineConfig,
# )


def Main():
    try:
        logger_train.info("Data_Ingestion: Data Ingestion Started")
        ## Strat-1
        # data_ings = DataIngestion(DataIngestionConfig(TrainingPipelineConfig()))

        ## Strat-2
        # train_pipe_config = TrainingPipelineConfig()
        # data_ings_config = DataIngestionConfig(train_pipe_config)
        # data_ings = DataIngestion(data_ings_config)

        ## Strat-3
        data_ings_artf = DataIngestion().init_data_ings()

        logger_train.info("Data_Ingestion: Data Ingestion Finished")

    except Exception as e:
        raise NetworkSecurityException(e)


if __name__ == "__main__":
    Main()
