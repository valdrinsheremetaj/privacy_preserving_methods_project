import os
import matplotlib.pyplot as plt
from config import UTILITY_METRICS, ATTACK_METRICS, RUNTIME_METRICS


def plot_results(results_df, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = results_df

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]

        baseline_df = model_df[model_df["mechanism"] == "none"]
        private_df = model_df[model_df["mechanism"] != "none"]

        for metric in UTILITY_METRICS + ATTACK_METRICS:
            plt.figure(figsize=(8, 5))

            std_col = f"{metric}_std"

            for mechanism in private_df["mechanism"].unique():
                mech_df = private_df[private_df["mechanism"] == mechanism]
                mech_df = mech_df.sort_values("epsilon")

                plt.errorbar(
                    mech_df["epsilon"],
                    mech_df[metric],
                    yerr=mech_df[std_col],
                    marker="o",
                    capsize=4,
                    label=mechanism,
                )

            # Non-private baseline as horizontal line
            if not baseline_df.empty:
                baseline_value = baseline_df[metric].mean()
                baseline_std = baseline_df[std_col].mean()

                plt.axhline(
                    y=baseline_value,
                    linestyle="--",
                    label="non-private baseline",
                )

                if baseline_std > 0:
                    plt.axhspan(
                        baseline_value - baseline_std,
                        baseline_value + baseline_std,
                        alpha=0.15,
                    )

            plt.xlabel("epsilon")
            plt.ylabel(metric)
            plt.title(f"{metric.upper()} vs epsilon - {model_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            file_name = f"{model_name}_{metric}_vs_epsilon.png"
            plt.savefig(os.path.join(output_dir, file_name), dpi=300)
            plt.close()

    plot_runtime(results_df, output_dir)


def plot_runtime(results_df, output_dir):
    runtime_summary = (
        results_df
        .groupby(["model", "mechanism"], as_index=False)
        .agg(
            training_runtime=("training_runtime", "mean"),
            training_runtime_std=("training_runtime", "std"),
            attack_runtime=("attack_runtime", "mean"),
            attack_runtime_std=("attack_runtime", "std"),
            total_runtime=("total_runtime", "mean"),
            total_runtime_std=("total_runtime", "std"),
        )
    )

    for std_col in [
        "training_runtime_std",
        "attack_runtime_std",
        "total_runtime_std",
    ]:
        runtime_summary[std_col] = runtime_summary[std_col].fillna(0)

    for model_name in runtime_summary["model"].unique():
        model_df = runtime_summary[runtime_summary["model"] == model_name]

        for metric in RUNTIME_METRICS:
            std_col = f"{metric}_std"

            plt.figure(figsize=(8, 5))

            plt.bar(
                model_df["mechanism"],
                model_df[metric],
                yerr=model_df[std_col],
                capsize=4,
            )

            plt.xlabel("mechanism")
            plt.ylabel("runtime seconds")
            plt.title(f"Average {metric.replace('_', ' ')} by mechanism - {model_name}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            file_name = f"{model_name}_{metric}_by_mechanism.png"
            plt.savefig(os.path.join(output_dir, file_name), dpi=300)
            plt.close()