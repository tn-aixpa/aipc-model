
import mlrun
import joblib
from zipfile import ZipFile
from sklearn import ensemble

@mlrun.handler()
def train_model(context, bikesharing_dataitem: mlrun.DataItem, model_metadata):
    reference = bikesharing_dataitem.as_df()
    target = 'target'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    
    params = model_metadata["parameters"]
    random_state = None
    n_estimators = None
    for param in params:
        if param["name"] == "random_state":
            random_state = param["value"]
        if param["name"] == "n_estimators":
            n_estimators = param["value"]
    print(n_estimators)
    regressor = ensemble.RandomForestRegressor(random_state = random_state, n_estimators = n_estimators)
    regressor.fit(reference[numerical_features + categorical_features], reference[target])
    joblib.dump(regressor, "bike_sharing_model.joblib")
    with ZipFile("bike_sharing_model.zip", "w") as z:
        z.write("bike_sharing_model.joblib")
    # log model to MLRun
    context.log_model(
        "bike_sharing_model",
        parameters={
            "random_state": random_state,
            "n_estimators": n_estimators
        },
        model_file="bike_sharing_model.zip",
        labels={"class": "ensemble.RandomForestRegressor("},
        algorithm="ensemble.RandomForestRegressor(",
        framework="ensemble"
    ) 
    
    
