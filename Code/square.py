import numpy as np


def square(y, X, w, order, index, Lambda=0.001):
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    w = np.asmatrix(w)
    n = X.shape[0]
    d = X.shape[1]

    if order == 0:
        Xw = np.asarray(X.dot(w))
        loss = np.linalg.norm(y - Xw, 2) ** 2

        return loss / n + Lambda * np.linalg.norm(w, 2) ** 2 / 2

    elif order == 1:
        Xw = np.asarray(X.dot(w))
        gradient = 2 * X.T.dot(Xw - y)

        return gradient / n + Lambda * w

    elif order == 2:

        hessian = 2 * X.T.dot(X)

        return hessian / n + Lambda * np.eye(d)

    elif order == 3:

        b = len(index)
        Xb = X[index, :]
        yb = y[index, :]

        Xwb = np.asarray(Xb.dot(w))
        stograd = 2 * Xb.T.dot(Xwb - yb)

        return stograd / b + Lambda * w

    elif order == 4:

        bH = len(index)
        Xb = X[index]

        sto_hessian = 2 * Xb.T.dot(Xb)

        return sto_hessian / bH + Lambda * np.eye(d)

    else:
        raise ValueError("The argument \"order\" should be 0, 1, 2, 3 or 4")