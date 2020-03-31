import numpy as np
import linereg
import matplotlib.pyplot as plt

# DATA
data = np.loadtxt("men-olympics-100.txt")

X, t = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))
t = t.reshape((len(t), 1))

lamdaValues = np.logspace(-8, 0, 100, base=10)

def LOOCV(X, t, l):
    errors = []
    for lam in lamdaValues:
        model = linereg.LinearRegression(lam=lam)
        modelError = 0
        for i in range(X.shape[0]):
            X_train = np.delete(X, i, 0)
            t_train = np.delete(t, i, 0)

            X_pred = X[i]
            X_pred = X_pred.reshape((-1, len(X_pred)))
            t_pred = t[i]
            t_pred = t_pred.reshape((len(t_pred), 1))

            model.fit(X_train, t_train)

            prediction = model.predict(X_pred)

            error = (prediction - t_pred)**2
            modelError += error[0][0]
        modelError = modelError / X.shape[0]
        errors.append(modelError)
    return errors

# First order polynomial
firstOrderLOOCV = LOOCV(X, t, lamdaValues)
bestValueindex = np.argmin(firstOrderLOOCV)
bestLamda = lamdaValues[bestValueindex]

model = linereg.LinearRegression(lam=bestLamda)
model.fit(X, t)
print("======= First order polynomial fitting =======")
print("The best value of lamda=%.10f and its loss=%.10f" %
      (bestLamda, firstOrderLOOCV[bestValueindex]))
print("For this model w0=%.10f and w1=%.10f \n" %
      (model.w[0][0], model.w[1][0]))

modelLam0 = linereg.LinearRegression()
modelLam0.fit(X, t)
print("For the model with lamda=0 w0=%.10f and w1=%.10f\n" %
      (modelLam0.w[0][0], modelLam0.w[1][0]))

# Plot data
plt.figure(figsize=(5,5))
plt.plot(lamdaValues, firstOrderLOOCV)
plt.title("LOOCV for different lamda values\n- First order polynomial fitting")
plt.xlabel("Lamda values")
plt.ylabel("Loss")
plt.show()

# Fourth order polynomial
def augment(X, max_order):
    X_augmented = X

    for i in range(2, max_order+1):
        X_augmented = np.concatenate([X_augmented, X**i], axis=1)
    return X_augmented


Xnew = augment(X, 4)
fourthOrderLOOCV = LOOCV(Xnew, t, lamdaValues)
bestValueindex = np.argmin(fourthOrderLOOCV)
bestLamda = lamdaValues[bestValueindex]

modelFourthOrder = linereg.LinearRegression(lam=bestLamda)
modelFourthOrder.fit(Xnew, t)
print("======= Fourth order polynomial fitting =======")
print("The best value of lamda=%.10f and its loss=%.10f" %
      (bestLamda, fourthOrderLOOCV[bestValueindex]))
print("For this model w0=%.10f, w1=%.10f, w2=%.10f, w3=%.10f, w4=%.10f \n" %
      (modelFourthOrder.w[0][0], modelFourthOrder.w[1][0],
       modelFourthOrder.w[2][0], modelFourthOrder.w[3][0],
    modelFourthOrder.w[4][0]))

# Plot data
plt.figure(figsize=(5,5))
plt.plot(lamdaValues, fourthOrderLOOCV)
plt.title("LOOCV for different lamda values\n- Fourth order polynomial fitting")
plt.xlabel("Lamda values")
plt.ylabel("Loss")
plt.show()