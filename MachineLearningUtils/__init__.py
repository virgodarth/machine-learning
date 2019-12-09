class BaseMachineLearning:
    """Base class for all learning algorithms
        Attributes
        ----------
        _model: Base model, default=None
            corresponding supervised machine learning algorithms Class
        model: default=None
            self._model after train
        _input_data: pd.DataFrame, default=None
            Feature data
        _output_data: pd.Series, default=None
            Target values
        input_data: dict, default={}
            Parameter setting that gave the best results on the hold out data.

        Parameters
        ----------
        input_data: pd.DataFrame, default=None
            Feature data
        output_data: pd.Series, default=None
            Target values
        mode: str, default=None
            output_data type. Choose classification or regression
    """

    def __init__(self, input_data, output_data=None, mode=None):
        if mode not in ['classification', 'regression']:
            raise ValueError('Invalid mode params. Have to choice classification or regression')

        self._input_data = input_data
        self._output_data = output_data

        self._mode = mode

        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = self.y_pred = self.y_prob = None
