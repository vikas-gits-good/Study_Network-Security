import os
import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import RedirectResponse

from Network_Security.Logging.logger_pred import logger_pred
from Network_Security.Exception.exception import NetworkSecurityException
from Network_Security.Pipeline.training_pipeline import TrainingPipeline
from Network_Security.Utils.main_utils.utils import load_object
from Network_Security.Constants.train_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
)
from Network_Security.Utils.ml_utils.model.estimator import NetworkModel

# Mongo DB Stuff. If you are allowing user to enter data in file/database format
ca = certifi.where()
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

db_client = MongoClient(host=mongo_db_url, tlsCAFile=ca)
database = db_client[DATA_INGESTION_DATABASE_NAME]
collection = db_client[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_ppln = TrainingPipeline().init_train_pipe()
        return Response("Training successfully completed")

    except Exception as e:
        raise NetworkSecurityException(e)


@app.post("/predict")
async def predict_route(request: Request = None, file: UploadFile = File(...)):
    try:
        logger_pred.info("Prediction_Pipeline: Prediction of user data started")

        logger_pred.info(
            "Prediction_Pipeline: Prediction Artifacts' Requisition started"
        )
        df = pd.read_csv(file.file)
        ppln_prpc = load_object("final_models/ppln_prpc.pkl")
        model = load_object("final_models/model.pkl")
        logger_pred.info(
            "Prediction_Pipeline: Prediction Artifacts' Requisition finished"
        )

        logger_pred.info("Prediction_Pipeline: Predicting User Data started")
        ntwk_modl = NetworkModel(pipeline=ppln_prpc, model=model)
        y_pred = ntwk_modl.predict(x=df)
        df["y_pred"] = y_pred
        logger_pred.info("Prediction_Pipeline: Predicting User Data finished")

        table_html = df.to_html(classes="table table-striped")
        df.to_csv("pred_output/output.csv")

        logger_pred.info("Prediction_Pipeline: Prediction of user data finished")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
