from artifact_types import Data, Configuration, Report
from holisticai.bias.mitigation import Reweighing

class DataPreparation():
    """ 
    Data preparation stage containing 4 operations
    Data Profiling
    Data Validation
    Data Preprocessing
    Data Documentation
    """
    def __init__(self):
        self.stage = "Data Preparation"
    
    
    def split_data_from_df(self, data):
        """Splits a DataFrame into features (X), labels (y), and demographic data (dem)."""
        y = data['Label'].values  # Extract labels
        X = data[[str(i) for i in np.arange(500)]].values  # Extract features
        filter_col = ['Ethnicity', 'Gender'] + [col for col in data if str(col).startswith('Ethnicity_')] + [col for col in data if str(col).startswith('Gender_')]  # Demographics
        dem = data[filter_col].copy()  # Extract demographics
        return X, y, dem  # Return features, labels, demographics

    
    def resample_equal(self, df, cat):
        """Resamples the DataFrame to balance categories by oversampling based on a combined category-label identifier."""
        df['uid'] = df[cat] + df['Label'].astype(str)  # Create unique identifier combining category and label
        enc = LabelEncoder()  # Initialize label encoder
        df['uid'] = enc.fit_transform(df['uid'])  # Encode the combined identifier
        res = imblearn.over_sampling.RandomOverSampler(random_state=6)  # Initialize oversampler
        df_res, euid = res.fit_resample(df, df['uid'].values)  # Apply oversampling
        df_res = pd.DataFrame(df_res, columns=df.columns)  # Convert to DataFrame
        df_res = df_res.sample(frac=1).reset_index(drop=True)  # Shuffle rows
        df_res['Label'] = df_res['Label'].astype(float)  # Convert label to float
        return df_res  # Return resampled DataFrame
    
    
    def bias_mitigation_pre_reweighing(self, data:Data) -> Data:
        """Reweighing is a pre-processing bias mitigation technique that amends the dataset to achieve statistical parity. 
        This method adjusts the weights of the samples in the dataset to compensate for imbalances between 
        different groups. By applying appropriate weights to each instance, 
        it ensures that the model is not biased towards any particular group, thereby promoting fairness. 
        The goal is to adjust the influence of each group so that the final model satisfies fairness criteria 
        such as statistical parity or disparate impact."""
        # Initialise and fit the Reweighing model to mitigate bias
        rew = Reweighing()

        # Define the groups (Black and White) in the training data based on the 'Ethnicity' column
        group_a_train = (dem_train['Ethnicity'] == 'Black')  # Group A: Black ethnicity
        group_b_train = (dem_train['Ethnicity'] == 'White')  # Group B: White ethnicity

        # Fit the reweighing technique to adjust sample weights
        rew.fit(y_train, group_a_train, group_b_train)

        # Extract the calculated sample weights from the reweighing model
        sample_weights = rew.estimator_params["sample_weight"]
        data_train['sample_weights'] = sample_weights
        
        return data_train
        