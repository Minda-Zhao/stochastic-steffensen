import numpy as np


def nonconvex(y, X, w, order, index, Lambda=0.001):
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    w = np.asmatrix(w)
    n = X.shape[0]
    d = X.shape[1]

    if order == 0:
        yXw = np.multiply(y, X.dot(w))
        loss = np.sum(1 - np.tanh(yXw))

        return loss / n + Lambda * np.linalg.norm(w, 2) ** 2 / 2

    elif order == 1:

        yXw = np.multiply(y, X.dot(w))
        weight = np.multiply(np.tanh(yXw), np.tanh(yXw)) - 1
        weight = np.multiply(weight, y)
        gradient = X.T.dot(weight)

        return gradient / n + Lambda * w

    elif order == 2:

        yXw = np.multiply(y, X.dot(w))
        hess_weight = 1 - np.multiply(np.tanh(yXw), np.tanh(yXw))
        hess_weight = np.multiply(hess_weight, y)
        hess_weight = np.multiply(hess_weight, y)
        hess_weight = 2 * np.multiply(hess_weight, np.tanh(yXw))
        hessian = np.multiply(hess_weight.T, X.T).dot(X) / n

        return hessian / n + Lambda * np.eye(d)

    elif order == 3:

        b = len(index)
        Xb = X[index, :]
        yb = y[index, :]

        yXwb = np.multiply(yb, Xb.dot(w))
        weight = np.multiply(np.tanh(yXwb), np.tanh(yXwb)) - 1
        weight = np.multiply(weight, yb)
        stograd = Xb.T.dot(weight)

        return stograd / b + Lambda * w

    elif order == 4:

        bH = len(index)
        Xb = X[index, :]
        yb = y[index, :]

        yXwb = np.multiply(yb, Xb.dot(w))
        hess_weight = 1 - np.multiply(np.tanh(yXwb), np.tanh(yXwb))
        hess_weight = np.multiply(hess_weight, yb)
        hess_weight = np.multiply(hess_weight, yb)
        hess_weight = 2 * np.multiply(hess_weight, np.tanh(yXwb))
        hessian = np.multiply(hess_weight.T, Xb.T).dot(Xb) / n

        return hessian / bH + Lambda * np.eye(d)

    else:
        raise ValueError("The argument \"order\" should be 0, 1, 2, 3 or 4")