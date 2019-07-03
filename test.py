import os

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from data_loading import create_data_transformer, load_dataset, load_data_from_file, Dataset
from models.neural_network import create_neural_network
from models.random_forest import RandomForestModel
from models.svm import create_svm

import numpy as np


def calculate_statistics(model, dataset):
    outputs = model(dataset.X)

    prediction = outputs > 0.5

    precision, recall, f1, _ = precision_recall_fscore_support(dataset.y, prediction)
    auc = roc_auc_score(dataset.y, prediction, average=None)
    return precision, recall, f1, auc


def print_statistics(model, dataset):
    precision, recall, f1, auc = calculate_statistics(model, dataset)

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('AUC:', auc)


def main():
    dir_name = '../dataset'  # todo: folder name through sys args
    transformer = create_data_transformer(os.path.join(dir_name, 'train.csv'))

    train, validate, test = load_dataset(dir_name, transformer)

    # train = Dataset(train.X[:100], train.y[:100])

    evaluation = load_data_from_file(os.path.join('../test-output/dataset', 'test.csv'), transformer)  # todo: folder name through sys args

    models = []
    for C in [0.5, 1., 1.5]: # [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]:
        print('Train for C =', C)
        models.append(create_svm(train, C))

    auc = [calculate_statistics(model, validate)[3] for model in models]
    print(auc)

    model = models[np.argmax(auc)]

    print('Train:')
    print_statistics(model, train)

    print()

    print('Test:')
    print_statistics(model, test)

    print()

    print('Evaluation:')
    print_statistics(model, evaluation)

    ## ...
    outputs = model(evaluation.X)
    prediction = outputs > 0.5

    for i in range(len(evaluation)):
        print(i, int(prediction[i]), evaluation.y[i])


if __name__ == '__main__':
    main()
