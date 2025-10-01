from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def select_classifier(model_name: str):
    """
    Select machine learning classifier based on model name

    Args:
        model_name (str): model name

    Returns:
        clf (sklearn.pipeline.Pipeline): classifier
    """
    if model_name == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=42)
    elif model_name == 'svm':
        clf = make_pipeline(
            SimpleImputer(strategy='mean'),
            MultiOutputClassifier(SVC(random_state=42))
        )
    elif model_name == 'naive_bayes':
        clf = make_pipeline(
            SimpleImputer(strategy='mean'),
            MultiOutputClassifier(GaussianNB())
        )
    elif model_name in ['mlp', 'neural_network', 'nn']:
        clf = make_pipeline(
            SimpleImputer(strategy='mean'),
            MultiOutputClassifier(
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=1e-4,
                    learning_rate='adaptive',
                    max_iter=300,
                    early_stopping=True,
                    random_state=42
                )
            )
        )
    elif model_name == 'random_forest':
        clf = RandomForestClassifier(random_state=42)
    elif model_name == 'gradient_boosting':
        clf = make_pipeline(
            SimpleImputer(strategy='mean'),
            MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' has not been implemented yet.")

    return clf
