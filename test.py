from utils import readArff, preprocess_data, accuracy_score, parse_filename
from AdaBoost import AdaBoost
from random import shuffle
import pandas as pd
import numpy as np

def k_fold_cross_val(filename, T, k):
    X, y = preprocess_data(readArff(filename))

    # Group examples s.t. fold sizes differ by at most 1.
    # Assign [N = floor(len(data) / k)] examples to each fold.
    # Assign one additional example to [r = len(data) % k] folds.
    groups_X = []
    groups_y = []
    size_per_group = len(X) // k
    r = len(X) % k
    start = 0
    for i in range(k):
        n_examples = size_per_group + (i < r)
        groups_X.append(X[start: start + n_examples])
        groups_y.append(y[start: start + n_examples])
        start += n_examples

    # cross validation each fold
    total_correct = total_data = 0
    for i in range(k):
        print("\nUsing group {} of {} as test data".format(i+1, k))
        train_data_X = np.array([x for group in groups_X[:i] + groups_X[i+1:] for x in group])
        train_data_y = np.array([x for group in groups_y[:i] + groups_y[i+1:] for x in group])
        test_data_X = np.array(groups_X[i])
        test_data_y = np.array(groups_y[i])

        ada = AdaBoost(T)
        ada.train(train_data_X, train_data_y)
        preds = ada.predict(test_data_X)
        n_correct = np.sum(preds == test_data_y, axis=0)
        n_data = len(preds)
        total_correct += n_correct
        total_data += n_data

    avg_acc = 100 * total_correct / total_data
    print("\nAverage accuracy: {:.2f}% ({}/{})"
          .format(avg_acc, total_correct, total_data))


def test(filename, T):
    X,y = preprocess_data(readArff(filename))
    ada = AdaBoost(T=T)
    ada.train(X,y)
    preds = ada.predict(X)
    _ = accuracy_score(preds, y, verbose=True)


if __name__ == '__main__':
    args = parse_filename()
    filename = args["filename"]
    k_folds = args["k_folds"]
    T = args["T"]


    if k_folds is None:
        print("\nTraining with entire dataset for train/test...")
        test(filename, T)
    else:
        print(f"\nTraining with {k_folds}-fold cross validation...")
        k_fold_cross_val(filename, T=T, k=k_folds)
