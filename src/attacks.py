import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from config import N_SHADOW_MODELS, SHADOW_TRAIN_SIZE, ATTACK_MAX_SAMPLES_PER_CLASS


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _as_array(y):
    return np.asarray(y)


def _confidence_scores(model, X, y):
    """
    Confidence of the model in the correct class.
    Used for the simple threshold-based membership inference attack.
    """
    probabilities = model.predict_proba(X)
    y_array = _as_array(y)

    return probabilities[np.arange(len(y_array)), y_array]


def _attack_features(model, X, y):
    """
    Features used by the shadow-model attack classifier.

    The attacker observes the target model's output behaviour:
    probabilities, confidence, entropy, loss, correctness, and the true label.
    """
    probabilities = model.predict_proba(X)
    y_array = _as_array(y)

    true_confidence = probabilities[np.arange(len(y_array)), y_array]
    max_confidence = probabilities.max(axis=1)

    entropy = -np.sum(
        probabilities * np.log(probabilities + 1e-12),
        axis=1,
    )

    predictions = model.predict(X)
    correctness = (predictions == y_array).astype(float)

    loss = -np.log(true_confidence + 1e-12)

    return np.column_stack([
        probabilities,
        true_confidence,
        max_confidence,
        entropy,
        loss,
        correctness,
        y_array,
    ])


def _balanced_target_samples(X_train, y_train, X_test, y_test, random_state, max_samples_per_class):
    """
    Build a balanced member/non-member attack set from the target model.
    Members come from the target training set, non-members from the test set.
    """
    rng = np.random.default_rng(random_state)

    n = min(len(X_train), len(X_test), max_samples_per_class)

    train_indices = rng.choice(len(X_train), size=n, replace=False)
    test_indices = rng.choice(len(X_test), size=n, replace=False)

    X_members = _take_rows(X_train, train_indices)
    y_members = _take_rows(y_train, train_indices)

    X_nonmembers = _take_rows(X_test, test_indices)
    y_nonmembers = _take_rows(y_test, test_indices)

    return X_members, y_members, X_nonmembers, y_nonmembers


def threshold_membership_attack(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state,
    max_samples_per_class=ATTACK_MAX_SAMPLES_PER_CLASS,
):
    """
    Simple threshold-based membership inference attack.

    A sample is predicted as a member if the model confidence in the true class
    is above a threshold. The threshold is selected to maximize attack accuracy
    on a balanced member/non-member attack set.
    """
    X_members, y_members, X_nonmembers, y_nonmembers = _balanced_target_samples(
        X_train,
        y_train,
        X_test,
        y_test,
        random_state,
        max_samples_per_class,
    )

    member_scores = _confidence_scores(model, X_members, y_members)
    nonmember_scores = _confidence_scores(model, X_nonmembers, y_nonmembers)

    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores)),
    ])

    thresholds = np.linspace(scores.min(), scores.max(), 200)

    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    attack_auc = roc_auc_score(labels, scores)

    return {
        "threshold_mia_accuracy": float(best_accuracy),
        "threshold_mia_auc": float(attack_auc),
    }


def shadow_membership_attack(
    target_model,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state,
    n_shadow_models=N_SHADOW_MODELS,
    shadow_train_size=SHADOW_TRAIN_SIZE,
    max_samples_per_class=ATTACK_MAX_SAMPLES_PER_CLASS,
):
    """
    Shadow-model membership inference attack.

    Shadow models are trained on auxiliary data from the same distribution.
    For each shadow model:
    - its own training samples are labelled as members
    - held-out samples are labelled as non-members

    A random forest attack classifier is then trained on the shadow model
    outputs and evaluated against the target model.
    """
    rng = np.random.default_rng(random_state)

    # Use test data as auxiliary data for the shadow models.
    # This avoids training shadow models directly on the target model's members.
    test_indices = rng.permutation(len(X_test))
    split = len(test_indices) // 2

    shadow_pool_indices = test_indices[:split]
    target_nonmember_indices = test_indices[split:]

    X_shadow_pool = _take_rows(X_test, shadow_pool_indices)
    y_shadow_pool = _take_rows(y_test, shadow_pool_indices)

    X_target_nonmember_pool = _take_rows(X_test, target_nonmember_indices)
    y_target_nonmember_pool = _take_rows(y_test, target_nonmember_indices)

    shadow_features = []
    shadow_labels = []

    pool_size = len(X_shadow_pool)
    per_shadow_needed = 2 * shadow_train_size
    replace = pool_size < per_shadow_needed

    for _ in range(n_shadow_models):
        if replace:
            chosen = rng.choice(pool_size, size=per_shadow_needed, replace=True)
        else:
            chosen = rng.choice(pool_size, size=per_shadow_needed, replace=False)

        train_idx = chosen[:shadow_train_size]
        test_idx = chosen[shadow_train_size:]

        X_shadow_train = _take_rows(X_shadow_pool, train_idx)
        y_shadow_train = _take_rows(y_shadow_pool, train_idx)

        X_shadow_test = _take_rows(X_shadow_pool, test_idx)
        y_shadow_test = _take_rows(y_shadow_pool, test_idx)

        shadow_model = clone(target_model)
        shadow_model.fit(X_shadow_train, y_shadow_train)

        member_features = _attack_features(shadow_model, X_shadow_train, y_shadow_train)
        nonmember_features = _attack_features(shadow_model, X_shadow_test, y_shadow_test)

        shadow_features.append(member_features)
        shadow_features.append(nonmember_features)

        shadow_labels.append(np.ones(len(member_features)))
        shadow_labels.append(np.zeros(len(nonmember_features)))

    X_attack_train = np.vstack(shadow_features)
    y_attack_train = np.concatenate(shadow_labels)

    attack_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        class_weight="balanced",
    )
    attack_model.fit(X_attack_train, y_attack_train)

    # Balanced target attack evaluation.
    n_target = min(len(X_train), len(X_target_nonmember_pool), max_samples_per_class)

    member_indices = rng.choice(len(X_train), size=n_target, replace=False)
    nonmember_indices = rng.choice(
        len(X_target_nonmember_pool),
        size=n_target,
        replace=False,
    )

    X_target_members = _take_rows(X_train, member_indices)
    y_target_members = _take_rows(y_train, member_indices)

    X_target_nonmembers = _take_rows(X_target_nonmember_pool, nonmember_indices)
    y_target_nonmembers = _take_rows(y_target_nonmember_pool, nonmember_indices)

    target_member_features = _attack_features(
        target_model,
        X_target_members,
        y_target_members,
    )
    target_nonmember_features = _attack_features(
        target_model,
        X_target_nonmembers,
        y_target_nonmembers,
    )

    X_attack_test = np.vstack([
        target_member_features,
        target_nonmember_features,
    ])
    y_attack_test = np.concatenate([
        np.ones(len(target_member_features)),
        np.zeros(len(target_nonmember_features)),
    ])

    attack_predictions = attack_model.predict(X_attack_test)
    attack_scores = attack_model.predict_proba(X_attack_test)[:, 1]

    attack_accuracy = accuracy_score(y_attack_test, attack_predictions)
    attack_auc = roc_auc_score(y_attack_test, attack_scores)

    return {
        "shadow_mia_accuracy": float(attack_accuracy),
        "shadow_mia_auc": float(attack_auc),
    }
