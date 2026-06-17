import os
import time
import pandas as pd
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from models import build_model
from evaluation import evaluate_model
from privacy_mechanisms import privatize_training_data, add_parameter_noise
from config import (
    FILE_PATH,
    MODELS,
    MECHANISMS,
    EPSILONS,
    DELTA,
    RANDOM_SEED,
    RESULTS_PATH,
    N_TRIALS,
    OVERFIT_TRAIN_SIZE,
    SUMMARY_METRICS,
)
from plotting import plot_results
from attacks import threshold_membership_attack, shadow_membership_attack


def make_seed(trial, model_idx, mechanism_idx, epsilon_idx):
    return (
        RANDOM_SEED
        + trial * 100000
        + model_idx * 10000
        + mechanism_idx * 1000
        + epsilon_idx
    )


def choose_training_data(X_train, y_train, model_name, trial, model_idx):
    """
    Select the training data for a model.

    The overfit stress-test model uses a fixed stratified subset per trial/model.
    This subset is independent of mechanism and epsilon, so privacy mechanisms
    are compared on the same training members.
    """
    if model_name != "overfit_mlp":
        return X_train, y_train

    stress_seed = RANDOM_SEED + trial * 100000 + model_idx * 10000

    X_small, _, y_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=OVERFIT_TRAIN_SIZE,
        stratify=y_train,
        random_state=stress_seed,
    )

    return X_small, y_small


def summarize_results(raw_results):
    aggregations = {}

    for metric in SUMMARY_METRICS:
        aggregations[metric] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    summary = (
        raw_results
        .groupby(["model", "mechanism", "epsilon"], as_index=False, dropna=False)
        .agg(**aggregations)
    )

    std_columns = [col for col in summary.columns if col.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)

    return summary


def print_dataset_info(X_train, X_test, y_train, y_test):
    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)


def print_summary(summary):
    print("\nAggregated results: mean ± std over trials\n")

    printable = summary.copy()

    for metric in SUMMARY_METRICS:
        printable[metric] = printable.apply(
            lambda row: f"{row[metric]:.4f} ± {row[metric + '_std']:.4f}",
            axis=1,
        )

    columns = ["model", "mechanism", "epsilon"] + SUMMARY_METRICS

    print(printable[columns].to_string(index=False))


def run_single_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    trial,
    model_name,
    model_idx,
    mechanism,
    mechanism_idx,
    epsilon,
    epsilon_idx,
):
    """
    Run one model + mechanism + epsilon combination.
    """

    seed = make_seed(trial, model_idx, mechanism_idx, epsilon_idx)

    total_start_time = time.perf_counter()

    training_start_time = time.perf_counter()

    X_train_used, y_train_used = choose_training_data(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        trial=trial,
        model_idx=model_idx,
    )

    X_train_private = privatize_training_data(
        X_train=X_train_used,
        mechanism=mechanism,
        epsilon=epsilon,
        random_state=seed,
    )

    model = build_model(model_name, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_private, y_train_used)
        
    model = add_parameter_noise(
        model=model,
        mechanism=mechanism,
        epsilon=epsilon,
        delta=DELTA,
        n_samples=len(X_train_private),
        random_state=seed,
    )

    training_runtime = time.perf_counter() - training_start_time

    metrics = evaluate_model(model, X_test, y_test)

    attack_start_time = time.perf_counter()

    threshold_attack_metrics = threshold_membership_attack(
        model=model,
        X_train=X_train_used,
        y_train=y_train_used,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
    )

    shadow_attack_metrics = shadow_membership_attack(
        target_model=model,
        X_train=X_train_used,
        y_train=y_train_used,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
    )

    attack_runtime = time.perf_counter() - attack_start_time
    total_runtime = time.perf_counter() - total_start_time

    return {
    "trial": trial,
    "seed": seed,
    "model": model_name,
    "mechanism": mechanism,
    "epsilon": epsilon,
    "training_runtime": training_runtime,
    "attack_runtime": attack_runtime,
    "total_runtime": total_runtime,

    "runtime": total_runtime,

    **metrics,
    **threshold_attack_metrics,
    **shadow_attack_metrics,
    }

def save_results(raw_results, summary):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    raw_path = RESULTS_PATH.replace(".csv", "_raw.csv")

    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(RESULTS_PATH, index=False)

    return raw_path


def get_epsilons(mechanism):
    if mechanism == "none":
        return [None]
    return EPSILONS


def run_experiments():
    result = preprocess_data(FILE_PATH)
    assert result is not None, "preprocess_data returned None"

    X_train, X_test, y_train, y_test = result
    print_dataset_info(X_train, X_test, y_train, y_test)

    rows = []

    for trial in range(N_TRIALS):
        print(f"\n=== Trial {trial + 1}/{N_TRIALS} ===")

        for model_idx, model_name in enumerate(MODELS):
            for mechanism_idx, mechanism in enumerate(MECHANISMS):
                for epsilon_idx, epsilon in enumerate(get_epsilons(mechanism)):

                    row = run_single_experiment(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        trial=trial,
                        model_name=model_name,
                        model_idx=model_idx,
                        mechanism=mechanism,
                        mechanism_idx=mechanism_idx,
                        epsilon=epsilon,
                        epsilon_idx=epsilon_idx,
                    )

                    rows.append(row)

                    eps_text = "baseline" if epsilon is None else epsilon
                    print(
                        f"trial={trial:<2} | {model_name:20s} | {mechanism:25s} | "
                        f"eps={eps_text} | acc={row['accuracy']:.4f} | "
                        f"f1={row['f1']:.4f} | auc={row['auc']:.4f} | "
                        f"train={row['training_runtime']:.2f}s | "
                        f"attack={row['attack_runtime']:.2f}s | "
                        f"total={row['total_runtime']:.2f}s"
                    )

    raw_results = pd.DataFrame(rows)
    summary = summarize_results(raw_results)

    raw_path = save_results(raw_results, summary)

    print_summary(summary)

    plot_results(summary, output_dir="results/plots")

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summarized results to: {RESULTS_PATH}")
    print("Saved plots to: results/plots/")

    return summary


if __name__ == "__main__":
    run_experiments()