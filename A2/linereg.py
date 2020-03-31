import numpy as np

class LinearRegression():
    def __init__(self, lam=0.0):

        self.lam = lam

    def fit(self, X, t):

        X = np.array(X).reshape((len(X), -1))
        t = np.array(t).reshape((len(t), 1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        print("X.shape: %i, t.shape: %i" % X.shape, t.shape) 
        diag = self.lam * len(X) * np.identity(X.shape[1])
        a = np.dot(X.T, X) + diag
        b = np.dot(X.T, t)
        self.w = np.linalg.solve(a,b)

    def predict(self, X):
        X = np.array(X).reshape((len(X), -1))
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        print("w.shape",self.w.shape )
        print("X.shape",X.shape )

        predictions = np.dot(X, self.w)

        return predictions
