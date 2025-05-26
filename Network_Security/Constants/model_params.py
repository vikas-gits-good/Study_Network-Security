from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

# from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier

models_dict = {
    # Ensemble Models
    "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=44),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        criterion="friedman_mse", random_state=44
    ),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=44),
    # "BaggingClassifier": BaggingClassifier(random_state=44, n_jobs=-1),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=44),
    "XGBClassifier": XGBClassifier(n_jobs=-1),
    "XGBRFClassifier": XGBRFClassifier(n_jobs=-1),
    # Tree Models
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=44),
    # "ExtraTreeClassifier": ExtraTreeClassifier(random_state=44),
    # Neightbour Models
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    # "RadiusNeighborsClassifier": RadiusNeighborsClassifier(n_jobs=-1),
    # Linear Models
    "LogisticRegression": LogisticRegression(n_jobs=-1),
    # "PassiveAggressiveClassifier": PassiveAggressiveClassifier(
    #     random_state=44
    # ),
    # Neural Network
    # "MLPClassifier": MLPClassifier(random_state=44),
}

params_dict = {
    "RandomForestClassifier": {
        # "criterion": ["gini", "entropy", "log_loss"],
        # "n_estimators": [200, 300, 400],
        # "max_depth": [8, 10],
        # "min_samples_split": [2, 3],
        # "min_samples_leaf": [2, 3],
        # "max_leaf_nodes": [30, 35],
        # "min_impurity_decrease": [0.001],
    },
    "GradientBoostingClassifier": {
        # "learning_rate": [0.1, 0.4, 0.5],
        # "n_estimators": [250, 300],
        # "min_samples_split": [2, 3],
        # "min_samples_leaf": [3, 4],
        # "max_depth": [5, 6],
        # "max_leaf_nodes": [5, 6],
    },
    "HistGradientBoostingClassifier": {
        # "learning_rate": [0.1, 0.4, 0.5],
        # "min_samples_leaf": [3, 4],
        # "max_depth": [5, 6],
        # "max_leaf_nodes": [5, 6],
    },
    # "BaggingClassifier": {
    # "n_estimators": [250, 300],
    # "max_features": [30, 40],
    # "max_samples": [350, 400],
    # },
    "AdaBoostClassifier": {
        # "n_estimators": [200, 250],
        # "learning_rate": [0.5, 1],
    },
    "XGBClassifier": {
        # "lambda": [1, 5, 10],
        # "alpha": [0, 5, 10],
    },
    "XGBRFClassifier": {
        # "lambda": [1, 5, 10],
        # "alpha": [0, 5, 10],
    },
    "DecisionTreeClassifier": {
        # "max_depth": [10, 12],
        # "min_samples_leaf": [10, 12],
        # "max_leaf_nodes": [5, 6],
    },
    # "ExtraTreeClassifier": {
    # "max_depth": [6, 8],
    # "min_samples_split": [2, 3],
    # "min_samples_leaf": [1, 2],
    # "max_leaf_nodes": [30, 40],
    # },
    "KNeighborsClassifier": {
        # "p": [1, 2],
        # "n_neighbors": [2, 3]
    },
    # "RadiusNeighborsClassifier": {
    # '':[],
    # },
    "LogisticRegression": {
        # "": [],
    },
    # "PassiveAggressiveClassifier": {
    # "": [],
    # },
    # "MLPClassifier": {
    #     # "": [],
    # },
}
