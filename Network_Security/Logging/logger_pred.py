import logging
import os
from datetime import datetime

# Define Log File Names
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGFILE_PRED = f"{timestamp}_pred.log"

# Define Log Folder paths
log_fold_pred = os.path.join(os.getcwd(), "logs", "pred")

# Create Log Folders if they don't exist
os.makedirs(log_fold_pred, exist_ok=True)

# Full log file paths
log_path_pred = os.path.join(log_fold_pred, LOGFILE_PRED)

# Define log format
log_fmt = "%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_fmt)

# Create logger_pred
logger_pred = logging.getLogger("logger_pred")
logger_pred.setLevel(logging.INFO)

if not logger_pred.hasHandlers():
    file_handler_pred = logging.FileHandler(log_path_pred)
    file_handler_pred.setFormatter(formatter)
    logger_pred.addHandler(file_handler_pred)
