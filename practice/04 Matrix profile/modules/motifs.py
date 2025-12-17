import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    mp = np.copy(matrix_profile['mp'])
    mpi = np.copy(matrix_profile['mpi'])
    excl_zone = matrix_profile['excl_zone']

    # Находим top-k мотивов
    for _ in range(top_k):
        # Находим индекс минимального значения (мотив - повторяющийся участок)
        min_idx = np.argmin(mp)
        min_val = mp[min_idx]

        # Проверка на NaN или Inf
        if is_nan_inf(min_val) or np.isinf(min_val):
            break

        # Индексы пары мотивов
        pair_idx = int(mpi[min_idx])
        motifs_idx.append((min_idx, pair_idx))
        motifs_dist.append(min_val)

        # Применяем зону исключения для обоих индексов
        mp = apply_exclusion_zone(mp, min_idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, pair_idx, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
