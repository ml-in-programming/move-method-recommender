from sklearn.preprocessing import StandardScaler

class PcaTransformer(object):
    def __init__(self, pca):
        self.__pca = pca

    def __call__(self, points):
        return self.__pca.transform(points)


class ScaleTransformer(object):
    def __init__(self, X):
        self.__scaler = StandardScaler().fit(X)

    def __call__(self, points):
        return self.__scaler.transform(points)
