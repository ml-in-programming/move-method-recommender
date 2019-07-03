from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, dataset):
        self.__model = RandomForestClassifier(n_estimators=200, max_depth=5)
        self.__model.fit(dataset.X, dataset.y)

    def __call__(self, points):
        return self.__model.predict(points)
