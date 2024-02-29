import io
import os
from datetime import datetime

import minio
from ultralytics import YOLO


# Recursively traverse the local folder and upload files.
def upload_files(folderPath, objectPrefix, minioClient, bucketName):
    # add date
    now = str(datetime.now())
    for root, _, files in os.walk(folderPath):
        for fileName in files:
            filePath = os.path.join(root, fileName)
            relativePath = os.path.relpath(filePath, folderPath)
            objectPath = os.path.join(now, objectPrefix, relativePath)

            minioClient.fput_object(bucketName, objectPath, filePath)


def yolo_train(context, params: dict):
    model = YOLO(params["base_model"])
    results = model.train(
        data=params["data"], epochs=params["epochs"], imgsz=params["imgsz"]
    )

    minioClient = minio.Minio(
        os.environ.get("MLRUN_K8S_SECRET__MINIO_URL"),
        access_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_AK"),
        secret_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_SK"),
    )

    # Set the bucket and folder names.
    bucketName = "models"
    folderName = "runs/detect/train"

    # Create the folder in Minio if it doesn't already exist.
    if not minioClient.bucket_exists(bucketName):
        minioClient.make_bucket(bucketName)

    # Upload files from the specified folder.
    upload_files(folderName, folderName, minioClient, bucketName)
