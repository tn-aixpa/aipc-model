from typing import List
import mlrun
import torch
import minio
from PIL import Image
import os


class ObjectDetector(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        torch.hub.set_dir(".")
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.minio_client = minio.Minio(
            os.environ.get("MLRUN_K8S_SECRET__MINIO_URL"),
            access_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_AK"),
            secret_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_SK"),
        )

    def dowload_file(self, bucket: str, path: str):
        try:
            # dowload image
            self.minio_client.fget_object(bucket, path, "test.jpg")
            im = Image.open("test.jpg")
            return im
        except Exception as e:
            print(e)

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        results = []
        bucket = body["bucket"]

        for path in body["inputs"]:
            image = self.dowload_file(bucket, path)
            result = self.model([image])
            results.append(result.tolist())

        return str(results)
