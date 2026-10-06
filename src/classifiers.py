"""Classifier configurations for the flow classification battery.

All classifiers use transport-layer features only. RandomForest and
GradientBoosting are limited in depth for clearer feature importance.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Wrap an estimator in a pipeline, optionally with StandardScaler."""
    if scale:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
    return Pipeline([("clf", estimator)])


CLASSIFIERS = {
    "logreg": _make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42),
        scale=True,
    ),
    "random_forest": _make_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "gradient_boosting": _make_pipeline(
        GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
    ),
    "gaussian_nb": _make_pipeline(
        GaussianNB(),
    ),
}


def get_classifier(name: str) -> Pipeline:
    """Get a fresh unfitted classifier pipeline by name.

    Args:
        name: One of: logreg, random_forest, gradient_boosting, gaussian_nb.

    Returns:
        Fresh sklearn Pipeline.

    Raises:
        ValueError: If name is not recognized.
    """
    if name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Valid: {', '.join(CLASSIFIERS.keys())}"
        )
    from sklearn.base import clone
    return clone(CLASSIFIERS[name])


CLASSIFIER_NAMES = list(CLASSIFIERS.keys())
