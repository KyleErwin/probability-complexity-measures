import numpy as np


def dist_matrix(df, target="target"):
    dist_dict = {}

    # Convert dataframe to NumPy array
    # Drop the 'target' column
    data = df.drop(columns=[target]).values
    data = data.astype(float)

    row_ids = df.index
    target_classes = df[target]

    # Iterate over each row in the dataframe
    # Exclude the last column (assuming it's the target class)
    for idx, row in df.drop(columns=[target]).iterrows():
        point = row.values.astype(float)
        distances = np.linalg.norm(data - point, axis=1)
        distances = list(zip(row_ids, target_classes, distances))
        distances.sort(key=lambda x: x[2])
        dist_dict[idx] = distances

    return dist_dict


def nearest_enemies(dist_matrix):
    enemies = set()
    for index in dist_matrix:
        _, target, _ = dist_matrix[index][0]

        for n in dist_matrix[index][1:]:
            n_index, n_target, _ = n

            if target != n_target:
                enemies.add(n_index)
                break

    return list(enemies)
