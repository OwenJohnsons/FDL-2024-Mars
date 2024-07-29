from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def get_classifier(classifier_name, params):
    if classifier_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif classifier_name == "LogisticRegression":
        return LogisticRegression(**params)
    elif classifier_name == "SVC":
        return SVC(**params)
    elif classifier_name == "KNeighbors":
        return KNeighborsClassifier(**params)
    elif classifier_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    elif classifier_name == "GradientBoosting":
        return GradientBoostingClassifier(**params)
    elif classifier_name == "AdaBoost":
        return AdaBoostClassifier(**params)
    elif classifier_name == "XGBoost":
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")