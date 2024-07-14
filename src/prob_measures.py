import numpy as np

from scipy.stats import gaussian_kde
from scipy.integrate import quad
from src.dist import *


def p1(df, col, target="target"):
    y = df["target"].to_numpy()
    n_classes = int(np.max(y) + 1)

    if n_classes == 2:
        class_a_data = df[df[target] == 0][col].to_numpy()
        class_b_data = df[df[target] == 1][col].to_numpy()
        return _pdf_shared_area(class_a_data, class_b_data)

    overlap = 0.0

    for i in range(n_classes):
        _copy = df.copy()
        one = _copy["target"] == i
        _all = (_copy["target"] > i) | (_copy["target"] < i)
        _copy.loc[one, "target"] = 1
        _copy.loc[_all, "target"] = 0
        overlap += p1(_copy, col, target)

    return overlap / n_classes


def p2(df):
    a, b = _ea_dists(df)
    class_a_data = np.array(a)
    class_b_data = np.array(b)
    return _pdf_shared_area_over_a(class_a_data, class_b_data)


def _pdf_shared_area_over_a(class_a_data, class_b_data):
    class_a_kde = gaussian_kde(class_a_data)
    class_b_kde = gaussian_kde(class_b_data)

    x_values = np.linspace(
        min(class_a_data.min(), class_b_data.min()),
        max(class_a_data.max(), class_b_data.max()),
        1000,
    )

    # Calculate the total area under PDFs
    total_area_class_a, _ = quad(
        lambda x: class_a_kde(x), x_values.min(), x_values.max()
    )

    # Calculate the shared area between PDFs
    shared_area, _ = quad(
        lambda x: min(class_a_kde(x), class_b_kde(x)), x_values.min(), x_values.max()
    )

    if total_area_class_a <= 0:
        return 1.0

    complexity = shared_area / total_area_class_a
    return complexity


def _pdf_shared_area(class_a_data, class_b_data):
    class_a_kde = gaussian_kde(class_a_data)
    class_b_kde = gaussian_kde(class_b_data)

    x_values = np.linspace(
        min(class_a_data.min(), class_b_data.min()),
        max(class_a_data.max(), class_b_data.max()),
        1000,
    )

    # Calculate the total area under PDFs
    total_area_class_a, _ = quad(
        lambda x: class_a_kde(x), x_values.min(), x_values.max()
    )
    total_area_class_b, _ = quad(
        lambda x: class_b_kde(x), x_values.min(), x_values.max()
    )

    # Calculate the shared area between PDFs
    shared_area, _ = quad(
        lambda x: min(class_a_kde(x), class_b_kde(x)), x_values.min(), x_values.max()
    )

    complexity = shared_area / (total_area_class_a + total_area_class_b - shared_area)
    return complexity


def _ea_dists(df):
    matrix = dist_matrix(df)

    ally_dists = []
    enemy_dists = []

    for idx in matrix:
        _, idx_class, _ = matrix[idx][0]
        for _, _class, ally_dist in matrix[idx][1:]:
            if idx_class == _class:
                break

        for _, _class, enemy_dist in matrix[idx][1:]:
            if idx_class != _class:
                break

        ally_dists.append(ally_dist)
        enemy_dists.append(enemy_dist)
    return ally_dists, enemy_dists
