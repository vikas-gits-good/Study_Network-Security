from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    trfm_object_file_path: str
    trfm_train_file_path: str
    trfm_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    score_f1: float
    score_precision: float
    score_recall: float


@dataclass
class ModelTrainerArtifact:
    trnd_modl_file_path: str
    train_metric_artf: ClassificationMetricArtifact
    test_metric_artf: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    local_artifact_dir: str
    local_model_dir: str
    cloud_artifact_dir: str
    cloud_model_dir: str
