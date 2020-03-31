import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")

X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
t_mean = np.mean(t_train)
print("The mean of the house prices in the training set is: %f" % t_mean)
tp = np.full((len(t_train), 1), t_mean)

# (b) RMSE function
def rmse(t, tp):
    return np.sqrt(((t - tp) ** 2).mean())

print("The RMSE between the true house prices and the estimates obtained via the simple \'mean\' model for the test set is: %f" % (rmse(t_test, tp)))

# (c) visualization of results
plt.figure(figsize=(5,5))
plt.scatter(t_test, tp)
plt.xlabel("True house prices")
plt.ylabel("Estimates")
plt.ylim([20,24])
plt.title("Evaluation")
plt.show()