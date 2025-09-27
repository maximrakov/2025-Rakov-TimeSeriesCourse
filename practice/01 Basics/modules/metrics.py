import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    # INSERT YOUR CODE
    if len(ts1) != len(ts2):
        raise ValueError("Временные ряды должны быть одинаковой длины")

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    # INSERT YOUR CODE
    if len(ts1) != len(ts2):
        raise ValueError("Временные ряды должны быть одинаковой длины")

        # Нормализуем временные ряды
    ts1_norm = (ts1 - np.mean(ts1)) / np.std(ts1)
    ts2_norm = (ts2 - np.mean(ts2)) / np.std(ts2)

    # Вычисляем евклидово расстояние между нормализованными рядами
    norm_ed_dist = ED_distance(ts1_norm, ts2_norm)

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    # INSERT YOUR CODE
    n = len(ts1)
    m = len(ts2)

    # Вычисляем размер окна warping window
    window_size = max(1, int(max(n, m) * r))

    # Создаем матрицу расстояний с бесконечностями
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0  # Начальная точка

    # Заполняем матрицу DTW
    for i in range(1, n + 1):
        # Определяем границы окна для j
        j_start = max(1, i - window_size)
        j_end = min(m + 1, i + window_size + 1)

        for j in range(j_start, j_end):
            # Вычисляем стоимость (евклидово расстояние в квадрате)
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            # Находим минимальный путь
            min_prev = min(dtw_matrix[i - 1, j],  # вставка в ts1
                           dtw_matrix[i, j - 1],  # вставка в ts2
                           dtw_matrix[i - 1, j - 1])  # соответствие

            dtw_matrix[i, j] = cost + min_prev

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist
