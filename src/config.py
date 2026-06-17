FILE_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
RESULTS_PATH = "results/privacy_mechanism_results.csv"

EPSILONS = [0.1, 0.5, 1.0, 5.0]
DELTA = 1e-5
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_TRIALS = 5

OVERFIT_TRAIN_SIZE = 2000
N_SHADOW_MODELS = 3
SHADOW_TRAIN_SIZE = 1000
ATTACK_MAX_SAMPLES_PER_CLASS = 5000

MODELS = ["logistic_regression", "mlp", "overfit_mlp"]

MECHANISMS = [
    "none",
    "laplace",
    "gaussian",
    "ldp_randomized_response",
]



UTILITY_METRICS = ["accuracy", "f1", "auc"]
ATTACK_METRICS = ["threshold_mia_accuracy", "threshold_mia_auc", "shadow_mia_accuracy", "shadow_mia_auc"]
RUNTIME_METRICS = [
    "training_runtime",
    "attack_runtime",
    "total_runtime",
]
SUMMARY_METRICS = UTILITY_METRICS + ATTACK_METRICS + RUNTIME_METRICS