
import json
import mlrun
import joblib
import pandas as pd
from zipfile import ZipFile

class BikeSharingDemandModel(mlrun.serving.V2ModelServer):
    def load(self):
        """Download and initialize the model """
        
        model_file, extra_data = self.get_model(".zip")
        files = ZipFile(model_file)
        files.extractall("/tmp/model")
        self.model = joblib.load(f"/tmp/model/{files.namelist()[0]}")

    def predict(self, body: dict):
        
        current = pd.DataFrame(body["inputs"])
        current_prediction = self.model.predict(current)
        response = json.loads(pd.Series(current_prediction).to_json(orient='records'))
        return response
        
