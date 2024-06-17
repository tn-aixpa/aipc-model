
import mlrun
import joblib
from zipfile import ZipFile
from sklearn import ensemble

@mlrun.handler()
def train_model(context, bikesharing_dataitem: mlrun.DataItem):
    reference = bikesharing_dataitem.as_df()
    target = 'target'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(reference[numerical_features + categorical_features], reference[target])
    joblib.dump(regressor, "bike_sharing_model.joblib")
    with ZipFile("bike_sharing_model.zip", "w") as z:
        z.write("bike_sharing_model.joblib")
    # log model to MLRun
    context.log_model(
        "bike_sharing_model",
        parameters={
            "random_state": 0,
            "n_estimators": 50
        },
        model_file="bike_sharing_model.zip",
        labels={"class": "ensemble.RandomForestRegressor("},
        algorithm="ensemble.RandomForestRegressor(",
        framework="ensemble"
    ) 
    
    
