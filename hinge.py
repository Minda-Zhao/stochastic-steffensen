import numpy as np


def hinge(y, X, w, order, index, Lambda):
    X = np.asmatrix(X)
    y = np.asmatrix(y).T
    w = np.asmatrix(w)
    n = X.shape[0]
    d = X.shape[1]

    if order == 0:
        yXw = np.multiply(y, X.dot(w))
        yXw_hinge = 1 - yXw
        yXw_hinge[yXw_hinge < 0] = 0
        loss = np.sum(np.multiply(yXw_hinge, yXw_hinge))
        # loss = np.sum(yXw_hinge)

        return loss / n + Lambda * np.linalg.norm(w, 2) ** 2 / 2

    elif order == 1:

        yXw = np.multiply(y, X.dot(w))
        sense = np.where(yXw < 1)[0]
        X_sense = X[sense, :]
        y_sense = y[sense, :]
        yXw_sense = np.multiply(y_sense, X_sense.dot(w))
        coef = - 2 * np.multiply(y_sense, 1 - yXw_sense)
        gradient = X_sense.T.dot(coef)

        # yXw = np.multiply(y, X.dot(w))
        # sense = np.where(yXw < 1)[0]
        # X_sense = X[sense, :]
        # y_sense = y[sense, :]
        # gradient = X_sense.T.dot(y_sense)

        return gradient / n + Lambda * w

    elif order == 2:

        yXw = np.multiply(y, X.dot(w))
        sense = np.where(yXw < 1)[0]
        X_sense = X[sense, :]
        hessian = 2 * X_sense.T.dot(X_sense)

        return hessian / n + Lambda * np.eye(d)

    elif order == 3:

        b = len(index)
        Xb = X[index, :]
        yb = y[index, :]
        yXwb = np.multiply(yb, Xb.dot(w))

        sense = np.where(yXwb < 1)[0]
        X_sense = Xb[sense, :]
        y_sense = yb[sense, :]
        yXw_sense = np.multiply(y_sense, X_sense.dot(w))

        coef = - 2 * np.multiply(y_sense, 1 - yXw_sense)
        stograd = X_sense.T.dot(coef)

        return stograd / b + Lambda * w

    elif order == 4:

        bH = len(index)
        Xb = X[index, :]
        yb = y[index, :]
        yXwb = np.multiply(yb, Xb.dot(w))

        sense = np.where(yXwb < 1)[0]
        X_sense = Xb[sense, :]

        sto_hessian = 2 * X_sense.T.dot(X_sense)

        return sto_hessian / bH + Lambda * np.eye(d)

    else:
        raise ValueError("The argument \"order\" should be 0, 1, 2, 3 or 4")