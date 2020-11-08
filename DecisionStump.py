import pandas as pd
import numpy as np

class DecisionStump:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_features = np.shape(self.X)[1]
        self.info_gain = None
        self.error = None
        self.best_attribute = None
        self.tree = dict()
        self.predictions = None
        self.class_val = 1

    def __str__(self):
        return f"""information_gain: {self.info_gain}, error: {self.error}, feature:{self.best_attribute}"""


    def _entropy(self, col):
        """
        Calculate the entropy with respect to the target column.
        """
        vals, counts = np.unique(col, return_counts = True)

        entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts))
                          for i in range(len(vals))])
        return entropy


    def _information_gain(self, attr):
        # calculate the entropy of the total dataset
        total_entropy = self._entropy(self.y)

        # calculate the sum of the weighted entropy of the attributes
        vals, counts = np.unique(attr, return_counts=True)


        weighted_entropy = sum([(counts[i]/np.sum(counts)) *
                            self._entropy(self.y[(attr == vals[i])]) for i in range(len(vals))])

        # calculate information gain
        info_gain = total_entropy - weighted_entropy
        return info_gain

    def _make_tree(self):
        # predict values based on self.best_attribute
        attr = self.X[:, self.best_attribute]
        vals, counts = np.unique(attr, return_counts=True)

        # tree = {attr_val1: p(1), attr_val2, p(1)}
        # keys represent branches, values represent probability of 1
        # we know the y's are {-1, 1}
        for val in vals:
            subset = self.y[(attr == val)]
            new_subset = np.where(subset == -1, 0, 1) # replace -1 with 0
            prob = sum(new_subset) / len(new_subset)
            self.tree[val] = prob

    def _predict(self):
        # predict values based on self.best_attribute
        attr = self.X[:, self.best_attribute]
        self.predictions = np.ones(np.shape(self.y))

        for i, x_i in enumerate(attr):
            if self.tree[x_i] < 0.5:
                self.predictions[i] = -1
            # if == 0.5 then could break tie with majority over everything -- add in at the end


    def _calculate_error(self):
        self._make_tree()
        self._predict()

        # calculate percent inaccuracy
        assert np.shape(self.predictions) == np.shape(self.y) # sanity check
        accuracy = np.sum(self.predictions == self.y, axis=0) / len(self.y)
        self.error = 1 - accuracy

    def learn(self):
        max_gain = float('-inf')

        for f in range(self.n_features):
            gain = self._information_gain(self.X[:, f])

            if max_gain < gain:
                self.info_gain = gain
                self.best_attribute = f
                max_gain = gain
        self._calculate_error()
