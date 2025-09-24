"""
Cross-validation utilities for offline evaluation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
)

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation utilities for model evaluation"""

    def __init__(self):
        self.cv_strategies = {
            "kfold": KFold,
            "stratified_kfold": StratifiedKFold,
            "time_series": TimeSeriesSplit,
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the cross validator"""
        pass

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup cross validator resources"""
        pass

    async def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_strategy: str = "kfold",
        n_splits: int = 5,
        scoring: str = "accuracy",
        return_train_score: bool = True,
    ) -> Dict[str, Any]:
    """Perform cross-validation on a model"""
        logger.info(
            f"Performing {cv_strategy} cross-validation with {n_splits} splits")

        # Select cross-validation strategy
        if cv_strategy == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif cv_strategy == "stratified_kfold":
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42)
        elif cv_strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unsupported CV strategy: {cv_strategy}")

        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            return_estimator=True,
        )

        # Calculate statistics
        results = {
            "cv_strategy": cv_strategy,
            "n_splits": n_splits,
            "scoring": scoring,
            "test_scores": cv_results["test_score"],
            "test_mean": np.mean(cv_results["test_score"]),
            "test_std": np.std(cv_results["test_score"]),
            "test_sem": np.std(cv_results["test_score"]) / np.sqrt(n_splits),
        }

        if return_train_score:
            results.update(
                {
                    "train_scores": cv_results["train_score"], "train_mean": np.mean(
                        cv_results["train_score"]), "train_std": np.std(
                        cv_results["train_score"]), "train_sem": np.std(
                        cv_results["train_score"]) / np.sqrt(n_splits), })

        # Calculate overfitting indicator
        if return_train_score:
            results["overfitting_indicator"] = (
                (results["train_mean"] -
                 results["test_mean"]) /
                results["test_std"] if results["test_std"] > 0 else 0)

        return results

    async def nested_cross_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        outer_cv: int = 5,
        inner_cv: int = 3,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
    """Perform nested cross-validation for hyperparameter tuning"""
        logger.info(
            f"Performing nested CV: {outer_cv} outer folds, {inner_cv} inner folds")

        from sklearn.model_selection import GridSearchCV, cross_val_score

        # Outer cross-validation
        outer_scores = []
        best_params_list = []

        outer_cv_folds = StratifiedKFold(
            n_splits=outer_cv, shuffle=True, random_state=42)

        for train_idx, test_idx in outer_cv_folds.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner cross-validation for hyperparameter tuning
            inner_cv_folds = StratifiedKFold(
                n_splits=inner_cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=inner_cv_folds,
                scoring=scoring,
                n_jobs=-1)

            grid_search.fit(X_train, y_train)
            best_params_list.append(grid_search.best_params_)

            # Evaluate on outer test set
            best_model = grid_search.best_estimator_
            score = best_model.score(X_test, y_test)
            outer_scores.append(score)

        return {
            "outer_scores": outer_scores,
            "outer_mean": np.mean(outer_scores),
            "outer_std": np.std(outer_scores),
            "best_params": best_params_list,
            "nested_cv_mean": np.mean(outer_scores),
            "nested_cv_std": np.std(outer_scores),
        }

    async def learning_curve_analysis(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: List[float] = None,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
    """Generate learning curve analysis"""
        from sklearn.model_selection import learning_curve

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        logger.info(
            f"Generating learning curve with {len(train_sizes)} training sizes")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1)

        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_scores_mean.tolist(),
            "train_scores_std": train_scores_std.tolist(),
            "val_scores_mean": val_scores_mean.tolist(),
            "val_scores_std": val_scores_std.tolist(),
            "learning_curve_data": {
                "train_scores": train_scores.tolist(),
                "val_scores": val_scores.tolist(),
            },
        }

    async def validation_curve_analysis(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: List[Any],
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
    """Generate validation curve analysis"""
        from sklearn.model_selection import validation_curve

        logger.info(f"Generating validation curve for parameter: {param_name}")

        train_scores, val_scores = validation_curve(
            model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        return {
            "param_name": param_name,
            "param_range": param_range,
            "train_scores_mean": train_scores_mean.tolist(),
            "train_scores_std": train_scores_std.tolist(),
            "val_scores_mean": val_scores_mean.tolist(),
            "val_scores_std": val_scores_std.tolist(),
            "validation_curve_data": {
                "train_scores": train_scores.tolist(),
                "val_scores": val_scores.tolist(),
            },
        }

    async def bootstrap_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 100,
        test_size: float = 0.2,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
    """Perform bootstrap validation"""
        from sklearn.model_selection import train_test_split

        logger.info(
            f"Performing bootstrap validation with {n_bootstrap} iterations")

        bootstrap_scores = []

        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_bootstrap, y_bootstrap, test_size=test_size, random_state=42
            )

            # Train and evaluate
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            bootstrap_scores.append(score)

        # Calculate confidence interval
        alpha = 0.05
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)

        return {
            "bootstrap_scores": bootstrap_scores,
            "mean_score": np.mean(bootstrap_scores),
            "std_score": np.std(bootstrap_scores),
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 1 - alpha},
        }
