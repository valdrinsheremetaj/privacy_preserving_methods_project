from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from config import RANDOM_SEED

def build_model(model_name):
    """
    Build a machine learning model based on the specified model name.

    Parameters:
    model_name (str): The name of the model to build. Supported values are "logistic_regression" and "decision_tree".

    Returns:
    Pipeline: A scikit-learn Pipeline object containing the specified model.
    """
    if model_name == "logistic_regression":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', 
             LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced", # important because of class imbalance between diabetes and no diabetes --> forces the model to pay more attention to the minority class (diabetes) during training
                    random_state=RANDOM_SEED
            ))
        ])
    elif model_name == "decision_tree":
        model = Pipeline([
            ('classifier', 
             DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=50,
                class_weight="balanced", # important because of class imbalance between diabetes and no diabetes --> forces the model to pay more attention to the minority class (diabetes) during training
                random_state=RANDOM_SEED
            ))
        ])
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported values are 'logistic_regression' and 'decision_tree'.")
    
    return model