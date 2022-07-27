import numpy as np


def logistic(y, X, w, order, index, Lambda=0.001):
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    w = np.asmatrix(w)
    n = X.shape[0]
    d = X.shape[1]

    if order == 0:
        Xw = np.asarray(X.dot(w))
        negative_exp_yXw = np.exp(-np.multiply(np.asarray(y), Xw))
        value = np.sum(np.log(1 + negative_exp_yXw)) / n + Lambda * np.linalg.norm(w, 2) ** 2 / 2

        return value

    elif order == 1:

        Xw = np.asarray(X.dot(w))

        # exp_yXw = np.exp(np.multiply(np.asarray(y), Xw))
        # grad_weight = np.asmatrix(np.divide(y, 1 + exp_yXw))

        yXw = np.multiply(y, Xw)
        grad_weight = .5 * (1 + np.tanh(.5 * -yXw))
        grad_weight = np.multiply(y, grad_weight)

        gradient = -1 * X.T.dot(grad_weight) / n + Lambda * w

        return gradient

    elif order == 2:

        Xw = np.asarray(X.dot(w))
        exp_yXw = np.exp(np.multiply(np.asarray(y), Xw))

        hess_weight = np.divide(exp_yXw, (1 + exp_yXw) ** 2)
        hess_weight = np.multiply(np.asarray(y) ** 2, hess_weight)
        hessian = np.multiply(hess_weight.T, X.T).dot(X) / n + Lambda * np.eye(d)

        return hessian

    elif order == 3:

        b = len(index)
        Xb = X[index, :]
        yb = y[index, :]
        Xwb = np.asarray(Xb.dot(w))

        # exp_yXwb = np.exp(np.multiply(np.asarray(yb), Xwb))
        # grad_weight = np.asmatrix(np.divide(yb, 1 + exp_yXwb))

        yXwb = np.multiply(yb, Xwb)
        grad_weight = .5 * (1 + np.tanh(.5 * -yXwb))
        grad_weight = np.multiply(yb, grad_weight)

        stograd = -1 * Xb.T.dot(grad_weight) / b + Lambda * w

        return stograd

    elif order == 4:

        bH = len(index)
        Xb = X[index]
        yb = y[index]

        Xwb = np.asarray(Xb.dot(w))
        exp_yXwb = np.exp(np.multiply(np.asarray(yb), Xwb))

        hess_weight = np.divide(exp_yXwb, (1 + exp_yXwb) ** 2)
        hess_weight = np.multiply(np.asarray(yb) ** 2, hess_weight)
        sto_hessian = np.multiply(hess_weight.T, Xb.T).dot(Xb) / bH + Lambda * np.eye(d)

        return sto_hessian

    else:
        raise ValueError("The argument \"order\" should be 0, 1, 2, 3 or 4")