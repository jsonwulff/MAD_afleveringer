import numpy as np
import lineweighreg
import matplotlib.pyplot as plt

# Load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")

# Choose all rows and all cols minus the last, choose all rows and the last col
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# Make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))

# Fit the model
model = lineweighreg.LinearWeightedRegression()
model.fit(X_train, t_train)

# Make predictions
predictions = model.predict(X_test)

# Plot the data
plt.figure(figsize=(5,5))
plt.scatter(t_test, predictions, s=3)
plt.title("A2")
plt.xlabel("True house price")
plt.ylabel("Estimate")
plt.show()

