from Network_Security.Exception.exception import NetworkSecurityException

from Network_Security.Entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score


def get_clasf_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        modl_f1_score = f1_score(y_true, y_pred)
        modl_precision_score = precision_score(y_true, y_pred)
        modl_recall_score = recall_score(y_true, y_pred)

        clasf_metric_artf = ClassificationMetricArtifact(
            score_f1=modl_f1_score,
            score_precision=modl_precision_score,
            score_recall=modl_recall_score,
        )
        return clasf_metric_artf

    except Exception as e:
        raise NetworkSecurityException(e)
