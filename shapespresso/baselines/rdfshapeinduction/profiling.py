import numpy as np
from scipy import stats


def min_cardinality(raw_cardinalities: list[int]):
    return np.min(raw_cardinalities)


def max_cardinality(raw_cardinalities: list[int]):
    return np.max(raw_cardinalities)


def cardinality_mean(raw_cardinalities: list[int]):
    return np.mean(raw_cardinalities)


def cardinality_mode(raw_cardinalities: list[int]):
    return stats.mode(raw_cardinalities)[0]


def cardinality_quadratic_mean(raw_cardinalities: list[int]):
    return np.sqrt(np.mean(np.square(raw_cardinalities)))


def cardinality_kurtosis(raw_cardinalities: list[int]):
    return stats.kurtosis(raw_cardinalities, fisher=True, bias=False)


def cardinality_standard_deviation(raw_cardinalities: list[int]):
    return np.std(raw_cardinalities, ddof=1)


def cardinality_skewness(raw_cardinalities: list[int]):
    return stats.skew(np.array(raw_cardinalities), bias=False)


def cardinality_variance(raw_cardinalities: list[int]):
    return np.var(raw_cardinalities)


def cardinality_percentile(raw_cardinalities: list[int], percentage: float):
    return np.percentile(raw_cardinalities, percentage)


def distinct_cardinalities_count(raw_cardinalities: list[int]):
    return len(set(raw_cardinalities))


def distinct_mean_cardinality(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return np.mean(distinct_cardinalities)


def distinct_quadratic_mean(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return np.sqrt(np.mean(np.square(distinct_cardinalities)))


def distinct_kurtosis(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return stats.kurtosis(distinct_cardinalities, fisher=True, bias=False)


def distinct_standard_deviation(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return np.std(distinct_cardinalities, ddof=1)


def distinct_skewness(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return stats.skew(np.array(distinct_cardinalities), bias=False)


def distinct_variance(raw_cardinalities: list[int]):
    distinct_cardinalities = list(set(raw_cardinalities))
    return np.var(distinct_cardinalities, ddof=1)


def percentages_min(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    return counts.min() / counts.sum()


def percentages_max(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    return counts.max() / counts.sum()


def zero_percentage(raw_cardinalities: list[int]):
    raw_cardinalities = np.array(raw_cardinalities)
    return np.mean(raw_cardinalities == 0)


def one_percentage(raw_cardinalities: list[int]):
    raw_cardinalities = np.array(raw_cardinalities)
    return np.mean(raw_cardinalities == 1)


def percentages_mean(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    return counts.mean() / counts.sum()


def percentages_quadratic_mean(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    percentages = counts / counts.sum()
    return np.sqrt(np.mean(np.square(percentages)))


def percentages_kurtosis(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    percentages = counts / counts.sum()
    return stats.kurtosis(percentages, fisher=True, bias=False)


def percentages_standard_deviation(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    percentages = counts / counts.sum()
    return np.std(percentages, ddof=1)


def percentages_skewness(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    percentages = counts / counts.sum()
    return stats.skew(percentages, bias=False)


def percentages_variance(raw_cardinalities: list[int]):
    values, counts = np.unique(raw_cardinalities, return_counts=True)
    percentages = counts / counts.sum()
    return np.var(percentages, ddof=1)


def feature_engineering(raw_cardinalities):
    features = [
        min_cardinality(raw_cardinalities),
        max_cardinality(raw_cardinalities),
        cardinality_mean(raw_cardinalities),
        cardinality_mode(raw_cardinalities),
        cardinality_quadratic_mean(raw_cardinalities),
        cardinality_kurtosis(raw_cardinalities),
        cardinality_standard_deviation(raw_cardinalities),
        cardinality_skewness(raw_cardinalities),
        cardinality_variance(raw_cardinalities),
        cardinality_percentile(raw_cardinalities, 98),
        cardinality_percentile(raw_cardinalities, 2),
        cardinality_percentile(raw_cardinalities, 75),
        cardinality_percentile(raw_cardinalities, 25),
        distinct_cardinalities_count(raw_cardinalities),
        distinct_mean_cardinality(raw_cardinalities),
        distinct_quadratic_mean(raw_cardinalities),
        distinct_kurtosis(raw_cardinalities),
        distinct_standard_deviation(raw_cardinalities),
        distinct_skewness(raw_cardinalities),
        distinct_variance(raw_cardinalities),
        percentages_min(raw_cardinalities),
        percentages_max(raw_cardinalities),
        zero_percentage(raw_cardinalities),
        one_percentage(raw_cardinalities),
        percentages_mean(raw_cardinalities),
        percentages_quadratic_mean(raw_cardinalities),
        percentages_kurtosis(raw_cardinalities),
        percentages_standard_deviation(raw_cardinalities),
        percentages_skewness(raw_cardinalities),
        percentages_variance(raw_cardinalities),
    ]
    return features
