from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Logging.logger_train import logger_train

from Network_Security.Constants.train_pipeline import (
    SAVED_MODEL_DIR,
    MODEL_TRAINER_BEST_MODEL_FILE_NAME,
)


# I donno which logger to put this in as both trai nand pred will use this function
class NetworkModel:
    def __init__(self, pipeline, model):
        try:
            self.ppln_prpc = pipeline
            self.model = model

        except Exception as e:
            raise NetworkSecurityException(e)

    def predict(self, x, y=None):
        try:
            x_trfm = self.ppln_prpc.transform(x)
            y_pred = self.model.predict(x_trfm)
            return y_pred

        except Exception as e:
            raise NetworkSecurityException(e)
