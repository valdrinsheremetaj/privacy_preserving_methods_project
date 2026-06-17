from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from config import RANDOM_SEED

def build_model(model_name, random_state=RANDOM_SEED):
    if model_name == "logistic_regression":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', 
             LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced", # important because of class imbalance between diabetes and no diabetes --> forces the model to pay more attention to the minority class (diabetes) during training
                    random_state=random_state
            ))
        ])
    elif model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=0.001,
                max_iter=100,
                random_state=random_state,
                early_stopping=True
            ))
        ])
    
    elif model_name == "overfit_mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0,
                max_iter=300,
                random_state=random_state,
                early_stopping=False
            ))
        ])

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported values are 'logistic_regression', 'mlp', and 'overfit_mlp'.")
    
    return model