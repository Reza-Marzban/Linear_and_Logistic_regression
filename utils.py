"""
Reza Marzban
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


def load_crash_data():
    data = np.loadtxt('crash.txt')
    training = data[0::2]
    validation = data[1::2]
    x_train = training[:, 0]
    y_train = training[:, 1]
    x_validation = validation[:, 0]
    y_validation = validation[:, 1]
    return x_train, y_train, x_validation, y_validation


def load_iris_data():
    def flower_to_float(s):
        d = {b'Iris-setosa': 0., b'Iris-versicolor': 1., b'Iris-virginica': 2.}
        return d[s]
    irises = np.loadtxt('iris.data', delimiter=',', converters = {4: flower_to_float})
    return irises


def preprocess_iris_data(irises):
    features = irises[:, :-1]
    labels = irises[:, -1]
    N = len(features)
    k = len(np.unique(labels))
    one_hot_labels = np.eye(k)[labels.astype(int)]
    features = np.append(features, np.ones(N).reshape(N, 1), 1)
    mid = N//2
    x_train = features[0::2]
    y_train = one_hot_labels[0::2]
    x_test = features[1::2]
    y_test = one_hot_labels[1::2]
    return x_train, y_train, x_test, y_test


class LinearRegressionPolynomial:

    l = list(range(1, 21))

    @staticmethod
    def basis_function(x, m):
        phi = np.array([x ** power for power in range(1, m+1)]).transpose()
        return phi

    def fit(self, x_train, y_train, x_validation, y_validation):
        train_rms = []
        validation_rms = []
        W = []
        for m in self.l:
            phi = self.basis_function(x_train, m)
            phi_T = phi.transpose()
            a = np.matmul(phi_T, phi)
            b = np.matmul(phi_T, y_train)
            w = np.linalg.solve(a, b)
            W.append(w)
            e = math.pow(np.linalg.norm(y_train-np.matmul(phi, w)), 2)/2
            N = len(x_train)
            rms = math.sqrt(2*e/N)
            train_rms.append(rms)
            phi_validation = self.basis_function(x_validation, m)
            e = math.pow(np.linalg.norm(y_validation-np.matmul(phi_validation, w)), 2)/2
            N = len(x_validation)
            rms_validation = math.sqrt(2*e/N)
            validation_rms.append(rms_validation)
        train_rms = np.array(train_rms)
        validation_rms = np.array(validation_rms)
        best_m = np.argmin(validation_rms)+1
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.title("RMS")
        plt.plot(self.l, train_rms, label="Training")
        plt.plot(self.l, validation_rms, label="Validation")
        plt.xticks(np.arange(1, 21, 1))
        plt.legend()
        x_test = np.linspace(x_train.min(), x_train.max())
        phi_test = self.basis_function(x_test, best_m)
        w = W[best_m-1]
        y_test = np.matmul(phi_test, w)
        plt.subplot(2, 1, 2)
        plt.title("Best Fit")
        plt.scatter(x_train, y_train, s=15, label="Training Data", color="darkblue")
        plt.scatter(x_validation, y_validation, s=15, label="Validation Data", color="darkgreen")
        plt.plot(x_test, y_test,"r", label="Validation Best Fit")
        plt.legend()
        plt.show()


class LinearRegressionRadial:

    l = [5, 10, 15, 20, 25]

    @staticmethod
    def basis_function(x, m):
        mu, deviation = np.linspace(x.min(), x.max(), m, retstep=True)
        phi = np.exp(np.array([((np.power(np.subtract(x, t), 2))/(-2*np.power(deviation, 2))) for t in mu]).transpose())
        return phi

    def fit(self, x_train, y_train, x_validation, y_validation):
        train_rms = []
        validation_rms = []
        W = []
        for m in self.l:
            phi = self.basis_function(x_train, m)
            phi_T = phi.transpose()
            a = np.matmul(phi_T, phi)
            b = np.matmul(phi_T, y_train)
            w = np.linalg.solve(a, b)
            W.append(w)
            e = math.pow(np.linalg.norm(y_train-np.matmul(phi, w)), 2)/2
            N = len(x_train)
            rms = math.sqrt(2*e/N)
            train_rms.append(rms)
            phi_validation = self.basis_function(x_validation, m)
            e1 = math.pow(np.linalg.norm(y_validation-np.matmul(phi_validation, w)), 2)/2
            N1 = len(x_validation)
            rms_validation = math.sqrt(2*e1/N1)
            validation_rms.append(rms_validation)
        train_rms = np.array(train_rms)
        validation_rms = np.array(validation_rms)
        best_m = np.argmin(validation_rms)+1
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.title("RMS")
        plt.plot(self.l, train_rms, label="Training")
        plt.plot(self.l, validation_rms, label="Validation")
        plt.xticks(np.arange(1, 21, 1))
        plt.legend()
        x_test = np.linspace(x_train.min(), x_train.max())
        phi_test = self.basis_function(x_test, self.l[best_m-1])
        w = W[best_m-1]
        y_test = np.matmul(phi_test, w)
        plt.subplot(2, 1, 2)
        plt.title("Best Fit")
        plt.scatter(x_train, y_train, s=15, label="Training Data", color="darkblue")
        plt.scatter(x_validation, y_validation, s=15, label="Validation Data", color="darkgreen")
        plt.plot(x_test, y_test,"r", label="Validation Best Fit")
        plt.legend()
        plt.show()


class LinearRegressionRadialWithPrior:
    l = 50
    beta = 0.0025
    alphas = np.logspace(-8, 0, 100)

    def basis_function(self, x):
        mu, deviation = np.linspace(x.min(), x.max(), self.l, retstep=True)
        phi = np.exp(
            np.array([((np.power(np.subtract(x, t), 2)) / (-2 * np.power(deviation, 2))) for t in mu]).transpose())
        return phi

    def fit(self, x_train, y_train, x_validation, y_validation):
        validation_rms = []
        W = []
        for alpha in self.alphas:
            alphaOverBeta = alpha/self.beta
            I = np.identity(self.l)
            I = alphaOverBeta*I
            phi = self.basis_function(x_train)
            phi_T = phi.transpose()
            a = np.matmul(phi_T, phi)+I
            b = np.matmul(phi_T, y_train)
            w = np.linalg.solve(a, b)
            W.append(w)
            phi_validation = self.basis_function(x_validation)
            e = math.pow(np.linalg.norm(y_validation-np.matmul(phi_validation, w)), 2)/2
            N = len(x_validation)
            rms_validation = math.sqrt(2*e/N)
            validation_rms.append(rms_validation)
        validation_rms = np.array(validation_rms)
        best_alpha_index = np.argmin(validation_rms)
        best_alpha = self.alphas[best_alpha_index]
        print(f"Best Alpha: {round(best_alpha,6)}")
        x_test = np.linspace(x_train.min(), x_train.max())
        phi_test = self.basis_function(x_test)
        w = W[best_alpha_index]
        y_test = np.matmul(phi_test, w)
        plt.figure(figsize=(12, 5))
        plt.title(f"Best Fit - Best Alpha: {round(best_alpha,6)}")
        plt.scatter(x_train, y_train, s=15, label="Training Data", color="darkblue")
        plt.scatter(x_validation, y_validation, s=15, label="Validation Data", color="darkgreen")
        plt.plot(x_test, y_test,"r", label="Validation Best Fit")
        plt.legend()
        plt.show()


class LogisticRegression:
    alpha = 0.003126
    w_init = np.ones(15)
    w_hat = None
    x_train = None
    y_train = None
    n = None
    k = None

    def _prior(self, w):
        prior = np.matmul(w.transpose(), w)*self.alpha/2
        return prior

    def _likelihood(self, w):
        likelihood = 0
        nominator = [self.y_train[:, k]*np.dot(w[5*k:(5*k+5)], np.transpose(self.x_train)) for k in range(self.k)]
        nominator = np.sum(np.transpose(np.array(nominator)), axis=-1)
        denominator = np.log(np.sum(np.transpose(np.exp([np.dot(w[5*k:(5*k+5)], np.transpose(self.x_train)) for k in range(self.k)])), axis=-1))
        likelihood += np.sum(nominator - denominator)
        return likelihood

    def _f(self, w):
        prior = self._prior(w)
        likelihood = self._likelihood(w)
        return prior-likelihood

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n = len(x_train)
        self.k = len(y_train[0])
        self.w_hat = minimize(self._f, self.w_init).x

    def classify_and_evaluate(self, x_test, y_test):
        if self.w_hat is None:
            print("Usage Error: Please use LogisticRegression.fit(x,y) first.")
            return
        w = self.w_hat
        k = self.k
        nominator = np.transpose(np.array([np.exp(np.dot(w[5*k:(5*k+5)], np.transpose(x_test))) for k in range(k)]))
        denominator = np.sum(np.transpose(np.exp([np.dot(w[5*k:(5*k+5)], np.transpose(x_test)) for k in range(k)])), axis=-1)
        s = np.transpose(np.array([nominator[:, k]/denominator for k in range(k)]))
        prediction = np.argmax(s, axis=-1)
        labels = np.array([np.where(y == 1)[0][0] for y in y_test])
        accuracy = np.sum(prediction == labels)/len(labels)
        return accuracy

if __name__ == "__main__":
    print("Usage: utils.py is a helper function for assn3.py!")