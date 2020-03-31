import numpy as np

class LinearWeightedRegression():
    def __init__(self):

        pass
    
    def fit(self, X, t):
        # Make sure that we have N-dimensional Numpy arrays (ndarray)
        X = np.array(X).reshape((len(X), -1))
        t = np.array(t).reshape((len(t), 1))

        # Prepend a column of ones til the X matrix
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        # Transform the w vector into a diagonal vector
        A = np.diag(t[:,0] ** 2)
       
        self.w = np.linalg.solve(X.T @ A @ X, X.T @ A @ t)

    def predict(self, X):
        X = np.array(X).reshape((len(X), -1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        predictions = np.dot(X, self.w)

        return predictions
