import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

SEED = 2334


def clean(model, scale=False):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)


def classifiers():
    models = {
        "majority": clean(DummyClassifier(strategy="most_frequent")),
        "logistic": clean(LogisticRegression(C=.1, class_weight="balanced", max_iter=3000), True),
        "decision_tree": clean(DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=SEED)),
        "random_forest": clean(RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=2)),
        "gbm": clean(GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=.05, random_state=SEED)),
        "svm": clean(SVC(C=2, class_weight="balanced", probability=True, random_state=SEED), True),
    }
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = clean(LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=8,
                                                  learning_rate=.04, verbosity=-1, class_weight="balanced", random_state=SEED, n_jobs=2))
    except (ImportError, OSError):
        pass
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = clean(XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.04, random_state=SEED, n_jobs=2))
    except (ImportError, OSError):
        pass
    return models


class RainflowCalibration(RegressorMixin, BaseEstimator):
    """Calibrate a power-law Miner proxy using training labels only."""
    def fit(self, X, y):
        self.columns_ = [f"rainflow_m{m}" for m in [1, 2, 3, 4, 5, 6, 8, 10]]
        self.powers_ = np.array([1, 2, 3, 4, 5, 6, 8, 10])
        logs = np.log(np.maximum(np.asarray(X[self.columns_], float), 1e-12))
        target = np.log(np.asarray(y))
        def loss(m):
            proxy = np.array([np.interp(m, self.powers_, row) for row in logs])
            intercept = np.median(target-proxy)
            return np.mean(np.abs(np.exp(proxy+intercept-target)-1))
        self.exponent_ = float(minimize_scalar(loss, bounds=(1, 10), method="bounded").x)
        proxy = np.array([np.interp(self.exponent_, self.powers_, row) for row in logs])
        self.intercept_ = float(np.median(target-proxy))
        return self

    def predict(self, X):
        logs = np.log(np.maximum(np.asarray(X[self.columns_], float), 1e-12))
        return np.exp([np.interp(self.exponent_, self.powers_, row)+self.intercept_ for row in logs])


def regressors():
    models = {"median": DummyRegressor(strategy="median"), "rainflow_calibration": RainflowCalibration(),
              "log_ridge": TransformedTargetRegressor(regressor=clean(Ridge(alpha=10), True), func=np.log, inverse_func=np.exp),
              "extra_trees": TransformedTargetRegressor(regressor=clean(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=2)), func=np.log, inverse_func=np.exp)}
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = TransformedTargetRegressor(regressor=clean(LGBMRegressor(n_estimators=120, num_leaves=5,
            min_child_samples=5, learning_rate=.04, verbosity=-1, random_state=SEED, n_jobs=2)), func=np.log, inverse_func=np.exp)
    except (ImportError, OSError):
        pass
    return models


def peer_scores(X, mode="thermal"):
    if mode == "car_order":
        return -np.arange(len(X), dtype=float)
    if mode == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        # Transductive peer outlier score, fitted only on this workbook without labels.
        values = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return -IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1).fit(values).score_samples(values)
    # Refrigerant undercharge primarily causes persistent excess cabin temperature.
    prefix = "Indoor Average Temperature"
    positive = np.maximum(X.get(prefix+"_peer_q90", 0), 0)
    mean = np.maximum(X.get(prefix+"_peer_mean", 0), 0)
    spread = np.maximum(X.get(prefix+"_peer_std", 0), 0)
    if mode == "thermal":
        return np.nan_to_num(np.asarray(mean + .5*positive, float))
    if mode == "robust":
        return np.nan_to_num(np.asarray(X.get(prefix+"_robust_z_q90", positive), float))
    return np.nan_to_num(np.asarray(mean + .5*positive + .5*spread, float))
