from river import tree


class OnlineModel:
    """
    Wrapper around River's Hoeffding Tree classifier.
    Responsible for:
    - Initializing model
    - Predicting
    - Learning
    - Resetting
    """

    def __init__(self):
        self._initialize_model()

    def _initialize_model(self):
        """
        Initializes a fresh model instance.
        """
        self.model = tree.HoeffdingTreeClassifier()

    def predict(self, x):
        """
        Predict one sample.
        """
        return self.model.predict_one(x)

    def learn(self, x, y):
        """
        Update model with one sample.
        """
        self.model.learn_one(x, y)

    def reset(self):
        """
        Hard reset of the model.
        """
        self._initialize_model()
