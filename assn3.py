"""
Reza Marzban
"""

from utils import load_crash_data, load_iris_data, preprocess_iris_data
from utils import LinearRegressionPolynomial, LinearRegressionRadial, LinearRegressionRadialWithPrior, LogisticRegression


def problem1():
    lrp = LinearRegressionPolynomial()
    x_train, y_train, x_validation, y_validation = load_crash_data()
    lrp.fit(x_train, y_train, x_validation, y_validation)


def problem2():
    lrr = LinearRegressionRadial()
    x_train, y_train, x_validation, y_validation = load_crash_data()
    lrr.fit(x_train, y_train, x_validation, y_validation)


def problem3():
    map = LinearRegressionRadialWithPrior()
    x_train, y_train, x_validation, y_validation = load_crash_data()
    map.fit(x_train, y_train, x_validation, y_validation)


def problem4():
    lr = LogisticRegression()
    irises = load_iris_data()
    x_train, y_train, x_test, y_test = preprocess_iris_data(irises)
    lr.fit(x_train, y_train)
    accuracy = lr.classify_and_evaluate(x_test, y_test)
    accuracy = round(accuracy, 4)
    print(f"Logistic Regression accuracy on test set: {accuracy}")


if __name__ == "__main__":
    # problem1()
    # problem2()
    # problem3()
    problem4()
