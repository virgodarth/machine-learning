from sklearn.model_selection import train_test_split


class BaseMachineLearning:
    def __init__(self, input_data, output_data, mode):
        self._input_data = input_data
        self._output_data = output_data
        self._mode = mode
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = self.y_predict = self.y_prob = None

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._input_data, self._output_data, test_size=test_size, random_state=random_state
        )

