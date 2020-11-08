import pandas as pd
import numpy as np
import argparse

def parse_filename():
    """
    Parses path/to/data.arff from command line.
    """
    parser = argparse.ArgumentParser(description="AdaBoost classifier",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("filename", type=str, help="path to .arff file")
    parser.add_argument("-K", "--k_folds", metavar="\b", type=int, default=None,
                        help="k-fold cross validation")

    parser.add_argument("-T", metavar="\b", type=int, default=20,
                        help="number of iterations")
    args = vars(parser.parse_args())
    return args

def readArff(filename):
    with open (filename, 'r') as f:
        # split lines, remove ones with comments
        lines = [line.lower() for line in f.read().split('\n') if not line.startswith('%')]

    # remove empty lines
    lines = [line for line in lines if line != '']

    columns = []
    data = []
    for index, line in enumerate(lines):
        if line.startswith('@attribute'):
            columns.append(line)

        if line.startswith('@data'):
            # get the rest of the lines excluding the one that says @data
            data = lines[index+1:]
            break

    # clean column names -- '@attribute colname  \t\t\t{a, b, ...}'
    cleaned_columns = [c[11:c.index('{')].strip() for c in columns]

    # clean and split data
    cleaned_data = [d.replace(', ', ',').split(',') for d in data]

    # create dataframe
    return pd.DataFrame(cleaned_data, columns = cleaned_columns)

def preprocess_data(df):
    # change class values to {-1, 1}
    y, unique = pd.factorize(df.iloc[:,-1])
    new_y = np.where(y==0, -1, 1)
    assert set(new_y) == {-1, 1}, 'Response variable must be ±1'

    # change xs to 2d numpy array
    xs = df.iloc[:,:-1]
    xs = xs.values

    return xs, new_y

def accuracy_score(y_true, y_pred, verbose=False):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    if verbose:
        print("Percent classified correctly: {:.2f}% ({}/{})"
              .format(100 * accuracy,
              np.sum(y_true == y_pred, axis=0),
              len(y_true)))
    return accuracy
