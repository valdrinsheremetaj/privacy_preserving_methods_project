import pandas as pd
import data_loader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_SEED


def preprocess_data(file_path):
    """
    Preprocess the data by loading it and performing necessary cleaning steps.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the preprocessed data.
    """
    # Load the data
    df = data_loader.load_data(file_path)
    
    if df is not None:
        # cut out duplicates, for a cleaner membership inference attack setup
        df = df.drop_duplicates()

        # binary diabetes vs no diabetes --> cut out prediabetes
        df = df[df["Diabetes_012"] != 1]

        X = df.drop(columns=["Diabetes_012"])
        y = (df["Diabetes_012"] == 2).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            stratify=y, # keeps same class distribution in train and test sets: important because of class imbalance between diabetes and no diabetes
            random_state=RANDOM_SEED
        )
        return X_train, X_test, y_train, y_test


    else:
        print("Data could not be loaded for preprocessing.")
        return None