# Privacy-Preserving Diabetes Classification

This project compares centralized and local privacy mechanisms against membership inference attacks on the CDC Diabetes Health Indicators dataset. The main task is binary diabetes classification. We train machine learning models with and without privacy mechanisms, then evaluate both model utility and privacy leakage.

The project was developed for the course **Privacy-Preserving Methods for Data Science and Distributed Systems** at the University of Basel.

## Project Idea

Machine learning models trained on sensitive health data can leak information about their training data. Even if the model does not directly store the dataset, an attacker may still infer whether a specific record was part of the training set. This is known as a membership inference attack.

In this project, we study this risk using the CDC Diabetes Health Indicators dataset. We compare normal non-private training with several privacy mechanisms and evaluate how much they reduce attack success.

## Dataset

The project uses the CDC Diabetes Health Indicators dataset based on BRFSS 2015 survey data.

Expected dataset path:

```text
data/diabetes_012_health_indicators_BRFSS2015.csv
```

The original target column is `Diabetes_012`, which contains three classes:

- `0`: no diabetes
- `1`: prediabetes
- `2`: diabetes

For this project, the task is simplified to binary classification. Prediabetes records are removed, and the target is converted to:

- `0`: no diabetes
- `1`: diabetes

Duplicates are also removed before the train-test split.

## Models

The implemented models are:

- Logistic Regression
- MLP
- Overfitting MLP

Logistic Regression is the main baseline model. The MLP is used as a non-linear model. The overfitting MLP is included as a stress test, because overfitting can make membership inference attacks stronger.

## Privacy Mechanisms

The following training settings are compared:

- `none`: normal non-private baseline
- `laplace`: Laplace parameter noise after training
- `gaussian`: Gaussian DP-style logistic regression
- `ldp_randomized_response`: local differential privacy using randomized response on input features

The privacy budgets are defined in `config.py`:

```python
EPSILONS = [0.1, 0.5, 1.0, 5.0]
DELTA = 1e-5
```

The Gaussian mechanism is implemented only for Logistic Regression.

## Membership Inference Attacks

Two attacks are implemented:

### Threshold-based attack

This attack uses the model confidence in the correct class. If the confidence is above a selected threshold, the sample is predicted as a member of the training set.

### Shadow-model attack

This attack trains auxiliary shadow models to imitate the target model. The outputs of these shadow models are used to train an attack classifier that predicts whether a sample was in the training set.

The attack features include:

- predicted probabilities
- confidence in the true class
- maximum confidence
- entropy
- loss
- correctness
- true label

## Evaluation Metrics

The project evaluates both utility and privacy leakage.

Utility metrics:

- accuracy
- F1-score
- AUC

Attack metrics:

- threshold MIA accuracy
- threshold MIA AUC
- shadow MIA accuracy
- shadow MIA AUC

Runtime metrics:

- training runtime
- attack runtime
- total runtime

## Project Structure

```text
.
├── attacks.py
├── config.py
├── data_loader.py
├── defenses.py
├── evaluation.py
├── models.py
├── plotting.py
├── preprocessing.py
├── run_experiments.py
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
└── results/
    ├── privacy_mechanism_results.csv
    ├── privacy_mechanism_results_raw.csv
    └── plots/
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Make sure the dataset is placed here:

```text
data/diabetes_012_health_indicators_BRFSS2015.csv
```

Then run:

```bash
python run_experiments.py
```

The script will:

1. load and preprocess the dataset,
2. train each model with each supported privacy mechanism,
3. evaluate model utility,
4. run the membership inference attacks,
5. save raw and summarized results,
6. generate plots.

## Output Files

The summarized results are saved to:

```text
results/privacy_mechanism_results.csv
```

The raw per-trial results are saved to:

```text
results/privacy_mechanism_results_raw.csv
```

Plots are saved to:

```text
results/plots/
```

Examples of generated plots:

- `logistic_regression_auc_vs_epsilon.png`
- `logistic_regression_f1_vs_epsilon.png`
- `logistic_regression_shadow_mia_auc_vs_epsilon.png`
- `overfit_mlp_shadow_mia_auc_vs_epsilon.png`
- `overfit_mlp_threshold_mia_auc_vs_epsilon.png`
- `logistic_regression_total_runtime_by_mechanism.png`

## Notes

The Laplace mechanism is used as an empirical parameter-noise baseline. It should not be interpreted as a complete formal differential privacy guarantee.

The Gaussian DP-style mechanism is implemented only for Logistic Regression because it depends on bounded inputs, L2 regularization, sensitivity estimation, and calibrated Gaussian noise.

The overfitting MLP is intentionally included to make membership leakage easier to observe and to test whether privacy mechanisms reduce this leakage.

## Authors

- Valdrin Sheremetaj
- Ashraf Jafarli
- Damaris Wahu
