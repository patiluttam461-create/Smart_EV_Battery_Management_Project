"""First-Order Takagi-Sugeno-Kang (TSK) ANFIS for Battery SOH Estimation.

Experimental model where rule consequents are linear affine functions of the
normalized input features plus an intercept:
    f_r(x) = a_{r, 1} x_1 + a_{r, 2} x_2 + ... + a_{r, D} x_D + b_r

Employs hybrid learning:
- Forward pass: Regularized Least-Squares Estimation (Ridge LSE) on centered targets.
- Backward pass: Adam gradient backpropagation for antecedent premise parameters (centers, spreads).
- Target centering ensures L2 shrinkage regularizes rule slopes and pulls offsets toward global mean SOH.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans


class FirstOrderSugenoANFIS(BaseEstimator, RegressorMixin):
    """First-Order Adaptive Neuro-Fuzzy Inference System.

    Parameters:
        n_rules: Number of fuzzy rules (clusters in scatter partitioning).
        lr: Learning rate for premise parameter gradient descent.
        lambda_reg: L2 regularization penalty for consequent LSE estimation.
        n_epochs: Maximum number of hybrid training epochs.
        sigma_min: Minimum allowable spread for Gaussian membership functions.
        random_state: Seed for reproducible cluster initialization.
    """

    def __init__(
        self,
        n_rules: int = 2,
        lr: float = 0.005,
        lambda_reg: float = 5.0,
        n_epochs: int = 100,
        sigma_min: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_rules = n_rules
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.sigma_min = sigma_min
        self.random_state = random_state

        # Fitted state
        self.y_mean_: float = 0.0
        self.centers_: Optional[np.ndarray] = None
        self.sigmas_: Optional[np.ndarray] = None
        self.consequents_: Optional[np.ndarray] = None  # Shape (n_rules, D + 1)
        self.history_: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}
        self.n_features_in_: int = 0

    @property
    def rule_count(self) -> int:
        """Return the number of fuzzy rules in the system."""
        return self.n_rules

    @property
    def parameter_count(self) -> Dict[str, int]:
        """Return counts of premise, consequent, and total parameters."""
        D = self.n_features_in_
        R = self.n_rules
        premise_count = 2 * R * D  # centers and sigmas
        consequent_count = R * (D + 1)  # linear weights + bias per rule
        return {
            "premise_parameters": premise_count,
            "consequent_parameters": consequent_count,
            "total_parameters": premise_count + consequent_count,
        }

    def _init_parameters(self, X: np.ndarray) -> None:
        """Initialize premise parameters via K-Means scatter clustering."""
        N, D = X.shape
        self.n_features_in_ = D

        kmeans = KMeans(
            n_clusters=self.n_rules,
            random_state=self.random_state,
            n_init=10,
        ).fit(X)
        self.centers_ = kmeans.cluster_centers_.copy()

        # Initialize Gaussian spreads based on pairwise inter-cluster distances
        dists = np.linalg.norm(
            self.centers_[:, None, :] - self.centers_[None, :, :], axis=-1
        )
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        avg_dist = float(np.mean(min_dists))
        init_sigma = max(avg_dist / np.sqrt(2.0), 0.5)
        self.sigmas_ = np.full_like(self.centers_, init_sigma)

        self.consequents_ = np.zeros((self.n_rules, D + 1))

    def _compute_normalized_weights(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalized rule firing strengths using log-sum-exp trick."""
        diff = X[:, None, :] - self.centers_[None, :, :]  # (N, R, D)
        log_mu = -0.5 * (diff / (self.sigmas_[None, :, :] + 1e-8)) ** 2
        log_w = np.sum(log_mu, axis=-1)  # (N, R)

        # Numerically stable softmax over rules
        max_log_w = np.max(log_w, axis=-1, keepdims=True)
        exp_w = np.exp(log_w - max_log_w)
        w_norm = exp_w / (np.sum(exp_w, axis=-1, keepdims=True) + 1e-12)
        return w_norm, log_w

    def _construct_design_matrix(
        self, X: np.ndarray, w_norm: np.ndarray
    ) -> np.ndarray:
        """Build linear design matrix M for first-order consequent estimation.

        For each sample i and rule r:
            A_{i, r} = w_norm[i, r] * [x_{i, 1}, x_{i, 2}, ..., x_{i, D}, 1]
        Output M is horizontal concatenation of A_{i, r} across all R rules.
        """
        N, D = X.shape
        R = self.n_rules
        X_ext = np.hstack([X, np.ones((N, 1))])  # (N, D+1)
        A_list = [w_norm[:, r : r + 1] * X_ext for r in range(R)]
        return np.hstack(A_list)  # (N, R * (D+1))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "FirstOrderSugenoANFIS":
        """Fit First-Order ANFIS using hybrid learning with target centering.

        Args:
            X: Training feature matrix of shape (N, D).
            y: Training target vector of shape (N,).
            X_val: Optional validation feature matrix.
            y_val: Optional validation target vector.

        Returns:
            self: Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        N, D = X.shape

        self.y_mean_ = float(np.mean(y))
        y_c = y - self.y_mean_

        self._init_parameters(X)
        self.history_ = {"train_rmse": [], "val_rmse": []}

        # Adam optimizer accumulators for premise parameters
        m_c = np.zeros_like(self.centers_)
        v_c = np.zeros_like(self.centers_)
        m_s = np.zeros_like(self.sigmas_)
        v_s = np.zeros_like(self.sigmas_)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, self.n_epochs + 1):
            w_norm, _ = self._compute_normalized_weights(X)
            M = self._construct_design_matrix(X, w_norm)

            # Step 1: Forward Pass - Regularized Least-Squares on Centered Target
            P = M.shape[1]
            A = M.T @ M + self.lambda_reg * np.eye(P)
            b = M.T @ y_c
            theta = np.linalg.solve(A, b)

            consequents_c = theta.reshape(self.n_rules, D + 1)
            # Add bias mean to intercept term of each rule
            self.consequents_ = consequents_c.copy()
            self.consequents_[:, -1] += self.y_mean_

            y_pred_c = M @ theta
            X_ext = np.hstack([X, np.ones((N, 1))])
            rule_outputs_c = np.array(
                [X_ext @ consequents_c[r] for r in range(self.n_rules)]
            ).T  # (N, R)

            y_pred = y_pred_c + self.y_mean_
            error = y_pred - y
            train_mse = float(np.mean(error**2))
            self.history_["train_rmse"].append(float(np.sqrt(train_mse)))

            # Step 2: Backward Pass - Premise Parameter Gradient Backpropagation
            f_diff = rule_outputs_c - y_pred_c[:, None]  # (N, R)
            dE_dz = (1.0 / N) * error[:, None] * w_norm * f_diff  # (N, R)
            diff = X[:, None, :] - self.centers_[None, :, :]  # (N, R, D)

            grad_c = np.sum(
                dE_dz[:, :, None] * (diff / (self.sigmas_[None, :, :] ** 2 + 1e-8)),
                axis=0,
            )
            grad_s = np.sum(
                dE_dz[:, :, None]
                * (diff**2 / (self.sigmas_[None, :, :] ** 3 + 1e-8)),
                axis=0,
            )

            # Gradient clipping to prevent numerical divergence
            grad_c = np.clip(grad_c, -5.0, 5.0)
            grad_s = np.clip(grad_s, -5.0, 5.0)

            # Adam updates for centers
            m_c = beta1 * m_c + (1.0 - beta1) * grad_c
            v_c = beta2 * v_c + (1.0 - beta2) * (grad_c**2)
            m_c_hat = m_c / (1.0 - beta1**epoch)
            v_c_hat = v_c / (1.0 - beta2**epoch)
            self.centers_ -= self.lr * m_c_hat / (np.sqrt(v_c_hat) + eps)

            # Adam updates for spreads
            m_s = beta1 * m_s + (1.0 - beta1) * grad_s
            v_s = beta2 * v_s + (1.0 - beta2) * (grad_s**2)
            m_s_hat = m_s / (1.0 - beta1**epoch)
            v_s_hat = v_s / (1.0 - beta2**epoch)
            self.sigmas_ -= self.lr * m_s_hat / (np.sqrt(v_s_hat) + eps)
            self.sigmas_ = np.clip(self.sigmas_, self.sigma_min, 10.0)

            # Validation checkpointing
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_mse = float(np.mean((val_pred - y_val) ** 2))
                self.history_["val_rmse"].append(float(np.sqrt(val_mse)))
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_state = (
                        self.centers_.copy(),
                        self.sigmas_.copy(),
                        self.consequents_.copy(),
                    )

        if best_state is not None:
            self.centers_, self.sigmas_, self.consequents_ = best_state

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate model predictions for input feature matrix X."""
        if self.centers_ is None or self.sigmas_ is None or self.consequents_ is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {X.shape[1]}"
            )

        w_norm, _ = self._compute_normalized_weights(X)
        M = self._construct_design_matrix(X, w_norm)
        return (M @ self.consequents_.reshape(-1)).ravel()
