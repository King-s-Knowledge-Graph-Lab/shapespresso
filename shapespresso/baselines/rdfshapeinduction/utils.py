import numpy as np


def cardinality_accuracy_score(y_true, y_pred):
    """
    Compute the (combined) cardinality accuracy score
    """
    return np.mean(np.all(y_true == y_pred, axis=1))
