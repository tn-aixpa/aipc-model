import json
import logging
import mlrun
import yaml
from os import path
from transformers import pipeline

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """Download and initialize the model and/or other elements"""
        logging.basicConfig(level=logging.INFO)
        # Load model metadata specifications
        model_metadata_path = "../../metadata/model.yml"
        with open(model_metadata_path, 'r') as model_md_content:
            models_metadata = yaml.safe_load(model_md_content)
        self.model_metadata = models_metadata[0]
        self.model_dir = self.model_metadata["training"]["output_dir"]
        self.model = self.load_model(self.model_dir)

    def predict(self, body: dict) -> list:
        """
        Make predictions
        """
        logging.basicConfig(level=logging.INFO)
        print(body)
        top_k = self.model_metadata["inference"]["parameters"]["top_k"]
        device = self.model_metadata["inference"]["parameters"]["device"]
        threshold = self.model_metadata["inference"]["parameters"]["threshold"]

        classifier = pipeline(
            "text-classification", 
            model=self.model_dir, 
            tokenizer=self.model_dir, 
            config=path.join(self.model_dir, "config.json"), 
            top_k=top_k, 
            device=device
            )
        text = body["inputs"][1]
        preds = classifier(text)

        predictions = []
        for pred in preds[0]:
            if pred["score"] > threshold:
                predictions.append({"label": pred["label"], "score": pred["score"]})
        return predictions
