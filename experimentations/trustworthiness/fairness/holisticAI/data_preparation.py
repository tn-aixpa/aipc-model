
class DataPreparation():
    def __init__(self):
        pass
    
    def split_data_from_df(self, data):
        """
        Splits a DataFrame into features (X), labels (y), and demographic data (dem).
        """
        y = data['Label'].values  # Extract labels
        X = data[[str(i) for i in np.arange(500)]].values  # Extract features
        filter_col = ['Ethnicity', 'Gender'] + [col for col in data if str(col).startswith('Ethnicity_')] + [col for col in data if str(col).startswith('Gender_')]  # Demographics
        dem = data[filter_col].copy()  # Extract demographics
        return X, y, dem  # Return features, labels, demographics


    def resample_equal(self, df, cat):
        """
        Resamples the DataFrame to balance categories by oversampling
        based on a combined category-label identifier.
        """
        df['uid'] = df[cat] + df['Label'].astype(str)  # Create unique identifier combining category and label
        enc = LabelEncoder()  # Initialize label encoder
        df['uid'] = enc.fit_transform(df['uid'])  # Encode the combined identifier
        res = imblearn.over_sampling.RandomOverSampler(random_state=6)  # Initialize oversampler
        df_res, euid = res.fit_resample(df, df['uid'].values)  # Apply oversampling
        df_res = pd.DataFrame(df_res, columns=df.columns)  # Convert to DataFrame
        df_res = df_res.sample(frac=1).reset_index(drop=True)  # Shuffle rows
        df_res['Label'] = df_res['Label'].astype(float)  # Convert label to float
        return df_res  # Return resampled DataFrame