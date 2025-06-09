from Network_Security.Logging.logger_train import logger_train
from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Utils.main_utils.utils import s3_sync
from Network_Security.Entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelPusherArtifact,
)
from Network_Security.Entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(
        self,
        model_train_artf: ModelTrainerArtifact = None,
        model_pushr_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        try:
            self.model_pushr_config = model_pushr_config
            self.model_train_artf = model_train_artf
        except Exception as e:
            raise NetworkSecurityException(e)

    def init_model_pusher(self):
        try:
            logger_train.info("Model_Pusher: Syncing Artifacts to S3 bucket started")
            s3_sync(
                source=self.model_pushr_config.lcl_artifact_dir,
                destination=self.model_pushr_config.url_artifact,
            )
            logger_train.info("Model_Pusher: Syncing Artifacts to S3 bucket finished")

            logger_train.info("Model_Pusher: Syncing Models to S3 bucket started")
            s3_sync(
                source=self.model_pushr_config.lcl_models_dir,
                destination=self.model_pushr_config.url_models,
            )
            logger_train.info("Model_Pusher: Syncing Models to S3 bucket finished")

            model_pushr_artf = ModelPusherArtifact(
                local_artifact_dir=self.model_pushr_config.lcl_artifact_dir,
                local_model_dir=self.model_pushr_config.lcl_models_dir,
                cloud_artifact_dir=self.model_pushr_config.url_artifact,
                cloud_model_dir=self.model_pushr_config.url_models,
            )
            return model_pushr_artf

        except Exception as e:
            raise NetworkSecurityException(e)
