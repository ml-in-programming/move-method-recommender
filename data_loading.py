import csv
import os

import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms

from transofrmers import PcaTransformer, ScaleTransformer


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)


def load_data_from_file(file_name, transformer):
    X, y = [], []
    with open(file_name) as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        for row in csv_reader:
            X.append(list(map(float, row[:-1])))

            if row[-1] == '1':
                y.append(1)
            else:
                y.append(0)  # todo: -1? or better through int()

    X = np.array(X)
    if transformer:
        X = transformer(X)

    return Dataset(X, y)


def load_dataset(dir_name, transformer):
    return load_data_from_file(os.path.join(dir_name, 'train.csv'), transformer),\
           load_data_from_file(os.path.join(dir_name, 'validate.csv'), transformer),\
           load_data_from_file(os.path.join(dir_name, 'test.csv'), transformer)


def create_pca_transformer(X, threshold):
    # todo: move to ctor of PcaTransformer

    pca = PCA()
    pca.fit(X)

    variance_ratio = list(filter(lambda x: x >= threshold, pca.explained_variance_ratio_))
    components_num = len(variance_ratio)
    explained_variance = sum(variance_ratio)

    print('Keep {} PCA components that explain {}% variance'.format(components_num, 100 * explained_variance))

    pca = PCA(n_components=components_num)
    pca.fit(X)

    return PcaTransformer(pca)


def create_data_transformer(data_file):
    train = load_data_from_file(data_file, None)

    X = train.X

    pca_transformer = create_pca_transformer(X, 0.001)
    X = pca_transformer(X)

    scale_transformer = ScaleTransformer(X)

    transformer = transforms.Compose([pca_transformer, scale_transformer])
    return transformer


def load_data(dir_name):
    transformer = create_data_transformer(os.path.join(dir_name, 'train.csv'))
    return load_dataset(dir_name, transformer)
