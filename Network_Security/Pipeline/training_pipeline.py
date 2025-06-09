from Network_Security.Logging.logger_train import logger_train
from Network_Security.Exception.exception import NetworkSecurityException

from Network_Security.Components.data_ingestion import DataIngestion
from Network_Security.Components.data_validation import DataValidation
from Network_Security.Components.data_transformation import DataTransformation
from Network_Security.Components.model_trainer import ModelTrainer
from Network_Security.Components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self):
        pass

    def init_train_pipe(self):
        try:
            logger_train.info("Data_Ingestion: Data Ingestion Started")
            data_ings_artf = DataIngestion().init_data_ings()
            logger_train.info("Data_Ingestion: Data Ingestion Finished")

            logger_train.info("Data_Validation: Data Validation Started")
            data_vald_artf = DataValidation(data_ings_artf).init_data_vald()
            logger_train.info("Data_Validation: Data Validation Finished")

            logger_train.info("Data_Transformation: Data Transformation Started")
            data_trfm_artf = DataTransformation(data_vald_artf).init_data_trfm()
            logger_train.info("Data_Transformation: Data Transformation Finished")

            logger_train.info("Model_Trainer: Model Training Started")
            model_train_artf = ModelTrainer(data_trfm_artf).init_model_train()
            logger_train.info("Model_Trainer: Model Training Finished")

            logger_train.info("Model_Pusher: Model Pusher Started")
            model_pusher_artf = ModelPusher(model_train_artf).init_model_pusher()
            logger_train.info("Model_Pusher: Model Pusher Finished")

            return model_train_artf

        except Exception as e:
            raise NetworkSecurityException(e)


if __name__ == "__main__":
    model_train_artf = TrainingPipeline().init_train_pipe()
