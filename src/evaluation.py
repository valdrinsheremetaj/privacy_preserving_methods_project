from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a machine learning model on the test set.

    Parameters:
    model (Pipeline): The trained machine learning model to evaluate.
    X_test (pd.DataFrame): The features of the test set.
    y_test (pd.Series): The true labels of the test set.

    Returns:
    dict: A dictionary containing the evaluation metrics (accuracy, F1 score, and AUC-ROC).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc)
    }