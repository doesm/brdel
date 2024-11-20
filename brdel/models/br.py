from sklearn.linear_model import BayesianRidge
from utils.data import featurize
from collections import namedtuple

Dataset = namedtuple("Dataset", ["train", "valid", "test"])
Example = namedtuple("Example", ["x", "y"])


class BayesianRidgeModel:
    def __init__(self, **hyperparameters):
        self.model = BayesianRidge(**hyperparameters)

    def prepare_dataset(self, df_train, df_valid, df_test, representation):
        X_train, y_train = self.featurize(df_train, representation)
        X_valid, y_valid = self.featurize(df_valid, representation)
        X_test, y_test = self.featurize(df_test, representation)
        self.data = Dataset(
            train=Example(x=X_train, y=y_train),
            valid=Example(x=X_valid, y=y_valid),
            test=Example(x=X_test, y=y_test),
        )
        return self.data

    def train(self):
        self.model.fit(self.data.train.x, self.data.train.y)

    def predict(self, x, return_uncertainty=False):
        preds = self.model.predict(x)
        if return_uncertainty:
            # Variance of predictions for uncertainty
            _, stds = self.model.predict(x, return_std=True)
            return preds, stds
        return preds

    def featurize(self, df, representation):
        return featurize(df, representation, smiles_col="smiles", label_col="y")
