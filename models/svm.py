from sklearn import svm
import numpy as np


class SvmModel:
    def __init__(self, model):
        self.__model = model

    def __call__(self, points):
        decisions = self.__model.decision_function(points)
        return 1 / (1 + np.exp(decisions * self.__model.probA_[0] + self.__model.probB_[0]))


def create_svm(dataset, C):
    #  todo: move to ctor of model
    model = svm.SVC(kernel='rbf', gamma='auto', C=C, probability=True)
    model.fit(dataset.X, dataset.y)

    return SvmModel(model)
