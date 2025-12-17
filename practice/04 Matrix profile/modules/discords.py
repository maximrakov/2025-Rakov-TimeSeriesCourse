import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE
    mp = np.copy(matrix_profile['mp'])
    mpi = np.copy(matrix_profile['mpi'])
    excl_zone = matrix_profile['excl_zone']

    # Находим top-k диссонансов
    for _ in range(top_k):
        # Диссонанс - элемент с максимальным значением матричного профиля
        max_idx = np.argmax(mp)
        max_val = mp[max_idx]

        # Проверка на NaN или Inf
        if is_nan_inf(max_val) or np.isinf(max_val):
            break

        # Индекс ближайшего соседа
        nn_idx = int(mpi[max_idx])

        # Сохраняем результаты
        discords_idx.append(max_idx)
        discords_dist.append(max_val)
        discords_nn_idx.append(nn_idx)

        # Применяем зону исключения, чтобы исключить пересечение с уже найденными диссонансами
        mp = apply_exclusion_zone(mp, max_idx, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
