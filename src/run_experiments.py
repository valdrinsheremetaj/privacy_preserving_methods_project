from preprocessing import preprocess_data
from models import build_model
from evaluation import evaluate_model
from config import MODELS, FILE_PATH


result = preprocess_data(FILE_PATH)
assert result is not None, "preprocess_data returned None"

X_train, X_test, y_train, y_test = result

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nClass balance train:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nClass balance test:")
print(y_test.value_counts(normalize=True).sort_index())

for model_name in MODELS:
    print(f"\nTesting model: {model_name}")

    model = build_model(model_name)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")