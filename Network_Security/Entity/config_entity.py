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


class DataValidationConfig:
    def __init__(
        self, train_pipe_config: TrainingPipelineConfig = TrainingPipelineConfig()
    ):
        self.data_valid_dir: str = os.path.join(
            train_pipe_config.artifact_dir, train_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_valid_dir, train_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_valid_dir, train_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, train_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, train_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, train_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, train_pipeline.TEST_FILE_NAME
        )

        self.drift_report_file_path: str = os.path.join(
            self.data_valid_dir,
            train_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            train_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(
        self, train_pipe_config: TrainingPipelineConfig = TrainingPipelineConfig()
    ):
        self.data_trfm_dir = os.path.join(
            train_pipe_config.artifact_dir, train_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        self.trfm_train_file_path: str = os.path.join(
            self.data_trfm_dir,
            train_pipeline.DATA_TRANSFORMATION_TRFM_DATA_DIR,
            train_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.trfm_test_file_path: str = os.path.join(
            self.data_trfm_dir,
            train_pipeline.DATA_TRANSFORMATION_TRFM_DATA_DIR,
            train_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.trfm_object_file_path: str = os.path.join(
            self.data_trfm_dir,
            train_pipeline.DATA_TRANSFORMATION_TRFM_OBJECT_DIR,
            train_pipeline.DATA_TRANSFORMATION_PIPELINE_OBJECT_FILE_NAME,
        )
