import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        dist_func = None

        # INSERT YOUR CODE
        if self.metric == 'euclidean':
            if self.is_normalize:
                dist_func = norm_ED_distance
            else:
                dist_func = ED_distance
        elif self.metric == 'dtw':
            dist_func = DTW_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Available options: 'euclidean', 'dtw'")

        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        # INSERT YOUR CODE

        dist_func = self._choose_distance()

        # Нормализуем данные если требуется
        if self.is_normalize and self.metric != 'euclidean':
            # Для неевклидовых метрик нормализуем вручную
            normalized_data = np.array([z_normalize(ts) for ts in input_data])
        else:
            normalized_data = input_data

        # Вычисляем попарные расстояния (только верхний треугольник)
        n = len(normalized_data)
        for i in range(n):
            for j in range(i + 1, n):  # Только верхний треугольник
                if self.metric == 'dtw':
                    # Для DTW передаем параметр r
                    distance = dist_func(normalized_data[i], normalized_data[j], r=1.0)
                else:
                    distance = dist_func(normalized_data[i], normalized_data[j])

                matrix_values[i, j] = distance
                matrix_values[j, i] = distance  # Симметрично заполняем нижний треугольник

        return matrix_values
