import os
from datetime import datetime
from Network_Security.Constants import train_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        self.pipeline_name = train_pipeline.PIPELINE_NAME
        self.artifact_name = train_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp = timestamp


class DataIngestionConfig:
    def __init__(
        self, train_pipe_config: TrainingPipelineConfig = TrainingPipelineConfig()
    ):
        self.data_ings_dir: str = os.path.join(
            train_pipe_config.artifact_dir, train_pipeline.DATA_INGESTION_INGESTED_DIR
        )
        self.feat_store_file_path = os.path.join(
            train_pipe_config.artifact_dir,
            train_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            train_pipeline.FILE_NAME,
        )
        self.train_file_path = os.path.join(
            self.data_ings_dir,
            train_pipeline.TRAIN_FILE_NAME,
        )
        self.test_file_path = os.path.join(
            self.data_ings_dir,
            train_pipeline.TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = (
            train_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = train_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = train_pipeline.DATA_INGESTION_DATABASE_NAME
