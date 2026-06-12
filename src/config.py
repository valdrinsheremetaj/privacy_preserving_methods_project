FILE_PATH = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"

EPSILONS = [0.1, 0.5, 1.0, 5.0]
DELTA = 1e-5
RANDOM_SEED = 42
TEST_SIZE = 0.2

MODELS = ["logistic_regression", "decision_tree"]

MECHANISMS = [
    "none",
    "laplace",
    "gaussian",
    "ldp_randomized_response",
]