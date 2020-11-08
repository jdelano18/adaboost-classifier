import pandas as pd
import numpy as np
from DecisionStump import DecisionStump

class AdaBoost:

    def __init__(self, T=20):
        self.T = T

    def _resample(self, X, y, w):
        # sanity checks
        assert sum(w) > 0.999999
        assert sum(w) < 1.000001

        assert np.shape(w)[0] == X.shape[0]

        # combine into dataframe
        xs = pd.DataFrame(X)
        df = pd.concat([xs, pd.DataFrame(y, columns=["y"])], axis=1)

        # resample
        new_data = df.sample(frac=1, replace=True, weights=w, random_state=1)
        new_xs = new_data.iloc[:,:-1].values
        new_ys = new_data.iloc[:, -1].values
        return new_xs, new_ys

    def train(self, X, y):
        epsilon = 1e-6 # add stability -- avoid div_by_0 error for when learner.error = 0

        n_instances = np.shape(X)[0]
        self.stumps = np.zeros(shape=self.T, dtype=object)
        self.alphas = np.zeros(shape=self.T)

        # initialize weights uniformly
        weights = np.ones(shape=n_instances) / n_instances
        # automatically use the whole data for first sample
        # since all weights are even
#         sample_X = X
#         sample_y = y

        for t in range(self.T):
            sample_X, sample_y = self._resample(X, y, weights)
            
            learner = DecisionStump(sample_X, sample_y)
            learner.learn()

            alpha = 0.5 * np.log((1 - learner.error + epsilon) / (learner.error + epsilon))

            weights *= np.exp(-alpha * sample_y * learner.predictions)
            weights /= weights.sum()

            # save stump objects and alphas
            self.stumps[t] = learner
            self.alphas[t] = alpha


    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))

        # For each classifier => label the samples
        for stump, alpha in zip(self.stumps, self.alphas):
            attr = X[:, stump.best_attribute]
            predictions = np.ones(np.shape(y_pred)) # Set all predictions to '1' initially

            for i, x_i in enumerate(attr):
                val = stump.tree.get(x_i, None) # guard against lookup not in tree for attribute
                if val is None or val < 0.5:
                    predictions[i] = -1 # switch predictions to -1 if p(1) < 0.5

            # Add predictions weighted by the classifiers alpha (alpha indicative of classifier's proficiency)
            y_pred += alpha * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred
