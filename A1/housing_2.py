import numpy as np
import pandas
import linreg
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# print("Initial X_train:",X_train.shape)
# print("Initial t_train:", t_train.shape)
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
# print("X_train after reshape", X_train.shape)
# print("X_train after reshape", t_train.shape)
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print("Single feature model weights w0 = %f and w1 = %f " % (model_single.w[0], model_single.w[1]))

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
print("Weights for all features model:")
print(model_all.w)

# (d) evaluation of results
def rmse(t, tp):
    return np.sqrt(((t - tp) ** 2).mean())

pred_single = model_single.predict(X_test[:,0])
rmse_single = rmse(t_test, pred_single)
print("RMSE of first feature only model: %f" % rmse_single)

plt.figure(figsize=(5,5))
plt.scatter(t_test, pred_single, s=3)
plt.xlabel("True house price")
plt.ylabel("Estimate")
plt.title("Single feature model")
plt.show()

pred_all = model_all.predict(X_test)
rmse_all = rmse(t_test, pred_all)
print("RMSE of all feature model: %f" % rmse_all)

plt.figure(figsize=(5,5))
plt.scatter(t_test, pred_all, s=3, color='red')
plt.xlabel("True house price")
plt.ylabel("Estimate")
plt.title("A1")
plt.show()




