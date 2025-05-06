import logging
import os
from datetime import datetime

# Define Log File Names
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGFILE_TRAIN = f"{timestamp}_train.log"

# Define Log Folder paths
log_fold_train = os.path.join(os.getcwd(), "logs", "train")

# Create Log Folders if they don't exist
os.makedirs(log_fold_train, exist_ok=True)

# Full log file paths
log_path_train = os.path.join(log_fold_train, LOGFILE_TRAIN)

# Define log format
log_fmt = "%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_fmt)

# Create logger_train
logger_train = logging.getLogger("logger_train")
logger_train.setLevel(logging.INFO)

# Avoid adding multiple handlers if logger_train already has handlers
if not logger_train.hasHandlers():
    file_handler_train = logging.FileHandler(log_path_train)
    file_handler_train.setFormatter(formatter)
    logger_train.addHandler(file_handler_train)
