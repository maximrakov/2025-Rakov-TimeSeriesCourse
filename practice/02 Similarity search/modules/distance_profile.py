import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    # INSERT YOUR CODE
    for i in range(N):
        # Берем подпоследовательность временного ряда
        subsequence = ts[i:i + m]

        if is_normalize:
            # Нормализуем и запрос, и подпоследовательность
            query_norm = z_normalize(query)
            subsequence_norm = z_normalize(subsequence)
            # Вычисляем расстояние между нормализованными рядами
            dist_profile[i] = ED_distance(query_norm, subsequence_norm)
        else:
            # Вычисляем расстояние без нормализации
            dist_profile[i] = ED_distance(query, subsequence)

    return dist_profile
