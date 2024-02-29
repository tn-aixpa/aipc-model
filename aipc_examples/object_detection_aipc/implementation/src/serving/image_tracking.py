from typing import List
import mlrun
from ultralytics import YOLO
import minio
import cv2
import os


class TrackingModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        self.minio_client = minio.Minio(
            os.environ.get("MLRUN_K8S_SECRET__MINIO_URL"),
            access_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_AK"),
            secret_key=os.environ.get("MLRUN_K8S_SECRET__MINIO_SK"),
        )
        self.model = self.load_model("models", "yolov8x.pt")

    def load_model(self, bucket, model_path):
        self.minio_client.fget_object(bucket, model_path, "yolov8.pt")
        model = YOLO("yolov8.pt")
        return model

    def dowload_file(self, bucket: str, path: str):
        try:
            # dowload image
            self.minio_client.fget_object(bucket, path, "test.jpg")
            im = cv2.imread("test.jpg")
            return im
        except Exception as e:
            print(e)

    def upload_results(self, bucket: str, path: str, image):
        try:
            cv2.imwrite("test.jpg", image)
            self.minio_client.fput_object(bucket, path, "test.jpg")
        except Exception as e:
            print(e)

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        bucket = body["bucket"]
        for path in body["inputs"]:
            image = self.dowload_file(bucket, path)
            results = self.model.track(image, persist=True)
            annotated_frame = results[0].plot()
            self.upload_results("video-tracking", path, annotated_frame)
