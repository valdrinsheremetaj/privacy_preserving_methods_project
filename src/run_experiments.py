import os
import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from attacks import shadow_membership_attack, threshold_membership_attack
from config import (
    DELTA,
    EPSILONS,
    FILE_PATH,
    MECHANISMS,
    MODELS,
    N_TRIALS,
    OVERFIT_TRAIN_SIZE,
    RANDOM_SEED,
    RESULTS_PATH,
    SUMMARY_METRICS,
)
from defenses import train_model_with_defense
from evaluation import evaluate_model
from plotting import plot_results
from preprocessing import preprocess_data


def make_seed(trial, model_idx, mechanism_idx, epsilon_idx):
    return (
        RANDOM_SEED
        + trial * 100000
        + model_idx * 10000
        + mechanism_idx * 1000
        + epsilon_idx
    )


def get_epsilons(mechanism):
    return [None] if mechanism == "none" else EPSILONS


def is_supported(model_name, mechanism):
    return not (mechanism == "gaussian" and model_name != "logistic_regression")


def choose_training_data(X_train, y_train, model_name, trial, model_idx):
    if model_name != "overfit_mlp":
        return X_train, y_train

    seed = RANDOM_SEED + trial * 100000 + model_idx * 10000
    X_small, _, y_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=OVERFIT_TRAIN_SIZE,
        stratify=y_train,
        random_state=seed,
    )
    return X_small, y_small


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
    seed = make_seed(trial, model_idx, mechanism_idx, epsilon_idx)
    total_start = time.perf_counter()

    X_used, y_used = choose_training_data(
        X_train,
        y_train,
        model_name,
        trial,
        model_idx,
    )

    train_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = train_model_with_defense(
            model_name=model_name,
            mechanism=mechanism,
            X_train=X_used,
            y_train=y_used,
            epsilon=epsilon,
            delta=DELTA,
            random_state=seed,
        )
    training_runtime = time.perf_counter() - train_start

    utility = evaluate_model(model, X_test, y_test)

    attack_start = time.perf_counter()
    threshold_attack = threshold_membership_attack(
        model=model,
        X_train=X_used,
        y_train=y_used,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
    )
    shadow_attack = shadow_membership_attack(
        target_model=model,
        X_train=X_used,
        y_train=y_used,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
    )
    attack_runtime = time.perf_counter() - attack_start
    total_runtime = time.perf_counter() - total_start

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
        **utility,
        **threshold_attack,
        **shadow_attack,
    }


def summarize_results(raw_results):
    agg = {}
    for metric in SUMMARY_METRICS:
        agg[metric] = (metric, "mean")
        agg[f"{metric}_std"] = (metric, "std")

    summary = (
        raw_results
        .groupby(["model", "mechanism", "epsilon"], as_index=False, dropna=False)
        .agg(**agg)
    )

    std_cols = [col for col in summary.columns if col.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0)
    return summary


def save_results(raw_results, summary):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    raw_path = RESULTS_PATH.replace(".csv", "_raw.csv")
    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(RESULTS_PATH, index=False)

    return raw_path


def print_dataset_info(X_train, X_test, y_train, y_test):
    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)


def print_row(row):
    eps = "baseline" if row["epsilon"] is None else row["epsilon"]
    print(
        f"trial={row['trial']:<2} | {row['model']:20s} | "
        f"{row['mechanism']:25s} | eps={eps} | "
        f"acc={row['accuracy']:.4f} | f1={row['f1']:.4f} | "
        f"auc={row['auc']:.4f} | train={row['training_runtime']:.2f}s | "
        f"attack={row['attack_runtime']:.2f}s | total={row['total_runtime']:.2f}s"
    )


def print_summary(summary):
    print("\nAggregated results: mean ± std over trials\n")

    table = summary.copy()
    for metric in SUMMARY_METRICS:
        table[metric] = table.apply(
            lambda row: f"{row[metric]:.4f} ± {row[metric + '_std']:.4f}",
            axis=1,
        )

    columns = ["model", "mechanism", "epsilon"] + SUMMARY_METRICS
    print(table[columns].to_string(index=False))


def run_experiments():
    X_train, X_test, y_train, y_test = preprocess_data(FILE_PATH)
    print_dataset_info(X_train, X_test, y_train, y_test)

    rows = []

    for trial in range(N_TRIALS):
        print(f"\n=== Trial {trial + 1}/{N_TRIALS} ===")

        for model_idx, model_name in enumerate(MODELS):
            for mechanism_idx, mechanism in enumerate(MECHANISMS):
                if not is_supported(model_name, mechanism):
                    print(f"Skipping unsupported combination: {model_name}, {mechanism}")
                    continue

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
                    print_row(row)

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
