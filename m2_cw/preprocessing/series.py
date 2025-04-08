import numpy as np


def get_evaluation_ids(N_val=50, N_test=50):
    np.random.seed(10022002)
    indices = np.arange(0, 1000, 1)
    val_ids = np.random.choice(indices, N_val, replace=False)
    indices = np.setdiff1d(indices, val_ids)
    test_ids = np.random.choice(indices, N_test, replace=False)

    return val_ids, test_ids


## I'll continue with this if I get time later. There are more important things
class Series():
    def __init__(self,
                 data: str | np.ndarray,
                 idx: int,
                 *args, **kwargs):
        
        self.data: np.ndarray = None
        self.id: int = None
        self.percentile: float = None
        self.scale_target: float = None
        self.scale_factor: float = None
        self.decimals: int = 2

        if isinstance(data, np.ndarray):
            self.from_numpy(data, *args, **kwargs)
        elif isinstance(data, str):
            self.from_string(data, *args, **kwargs)
        else:
            raise TypeError(...)

        if isinstance(idx, int):
            self.id = idx
        else:
            raise TypeError(...)

        if kwargs["decimals"]:
            try:
                assert isinstance(kwargs["decimals"], int)
            except AssertionError:
                raise TypeError(...)
            try:
                assert kwargs["decimals"] > 0
            except AssertionError:
                raise ValueError(...)
            self.decimals = kwargs["decimals"]
    
    def from_numpy(self, data_array: np.ndarray, *args, **kwargs):
        try:
            if kwargs["is_scaled"]:
                # Initialise the series from prescaled array
                try:
                    # We need the divisor used to scale the data
                    self.scale_factor = kwargs["scale_factor"]
                    self.percentile = kwargs["percentile"]
                    self.scale_target = np.percentile(data_array, self.percentile)

                except:
                    raise TypeError(...)
            else:
                # Initialise the series from unscaled array
                try:
                    if kwargs["alpha"] and kwargs["decimals"]:
                        ...
                except:
                    ...
        except IndexError:
            raise TypeError(...)

    
    def from_string(self, data_string: str):
        ...
    
    def to_numpy(self):
        ...
    
    def to_string(self):
        ...

    
class TrainSeries(Series):
    def __init__(self):
        super(TrainSeries, self).__init__()


class EvalSeries(Series):
    def __init__(self):
        ...