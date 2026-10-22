"""Machine Learning Baseline Models for EV Battery SOH Estimation.

Implements benchmark baselines:
1. Random Forest Regressor (Ensemble Decision Trees)
2. Multi-layer Perceptron Regressor (Artificial Neural Network / MLP)
"""

from typing import Any, Dict
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def get_random_forest_baseline(
    n_estimators: int = 50,
    max_depth: int = 12,
    min_samples_split: int = 10,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Create a configured Random Forest Regressor baseline.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
        random_state: Seed for reproducible ensemble generation.

    Returns:
        Configured RandomForestRegressor.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )


def get_mlp_baseline(
    hidden_layer_sizes: tuple = (64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 0.01,
    max_iter: int = 1000,
    random_state: int = 42,
) -> TransformedTargetRegressor:
    """Create a configured Multi-layer Perceptron (ANN) baseline.

    Uses TransformedTargetRegressor with StandardScaler to ensure neural
    network gradient updates are scale-invariant with respect to SOH %.

    Args:
        hidden_layer_sizes: Architecture of hidden layers.
        alpha: L2 penalty parameter.
        learning_rate_init: Initial learning rate for Adam.
        max_iter: Maximum optimization iterations.
        random_state: Seed for weight initialization.

    Returns:
        TransformedTargetRegressor wrapping MLPRegressor.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=random_state,
    )
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


def get_model_parameter_count(model: Any) -> int:
    """Calculate the trainable / active parameter count for a given model."""
    if hasattr(model, "parameter_count"):
        return model.parameter_count["total_parameters"]

    # For TransformedTargetRegressor, extract underlying regressor
    regressor = getattr(model, "regressor_", getattr(model, "regressor", model))

    if isinstance(regressor, MLPRegressor):
        if hasattr(regressor, "coefs_") and hasattr(regressor, "intercepts_"):
            n_params = sum(w.size for w in regressor.coefs_) + sum(
                b.size for b in regressor.intercepts_
            )
            return int(n_params)
        return 0

    if isinstance(regressor, RandomForestRegressor):
        if hasattr(regressor, "estimators_"):
            n_nodes = sum(tree.tree_.node_count for tree in regressor.estimators_)
            return int(n_nodes)
        return 0

    return 0
