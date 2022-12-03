import numpy as np
import time


def backtracking(func, x, direction, alpha=0.4, beta=0.9, maximum_iterations=100):
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if alpha >= 0.5:
        raise ValueError("Alpha must be less than 0.5")
    if beta <= 0:
        raise ValueError("Beta must be positive")
    if beta >= 1:
        raise ValueError("Beta must be less than 1")

    x = np.asarray(x)
    direction = np.asarray(direction)

    value = func(x, 0)
    gradient = func(x, 1)
    value = np.double(value)
    gradient = np.asarray(gradient)

    derivative = np.vdot(direction, gradient)

    # checking that the given direction is indeed a descent direction
    if derivative >= 0:
        return 0

    else:
        t = 1
        iterations = 0
        while True:

            if np.double(func(x + t * direction, 0)) < value + alpha * t * derivative:
                break

            t *= beta

            iterations += 1
            if iterations >= maximum_iterations:
                t = 0
                break

        return t


def gradient_descent(func, initial_x, eps, maximum_iterations, linesearch, *linesearch_args):
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.asarray(initial_x.copy())

    # initialization
    values = []
    xs = []
    iterations = 0

    # gradient updates
    while True:

        value = func(x, 0)
        gradient = func(x, 1)
        value = np.double(value)
        gradient = np.asarray(gradient)

        print(value)

        # updating the logs
        values.append(value)
        xs.append(x.copy())

        direction = - gradient

        t = linesearch(func, x, direction, *linesearch_args)

        if (np.vdot(gradient, gradient) <= eps) | (t == 0):
            break

        x += t * direction

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")

    return x, values


def newton(func, initial_x, eps, maximum_iterations, linesearch, *linesearch_args):
    x = np.asarray(initial_x.copy())

    # initialization
    values = []
    iterations = 0

    # newton's method updates
    while True:

        value = func(x, 0)
        gradient = func(x, 1)
        hessian = func(x, 2)
        value = np.double(value)
        gradient = np.asmatrix(gradient)
        hessian = np.asmatrix(hessian)

        # updating the logs
        values.append(value)

        direction = -1 * np.linalg.solve(hessian, gradient)

        newton_decrement = np.sqrt(gradient.T.dot(-1 * direction))

        t = linesearch(func, x, direction, *linesearch_args)

        if (newton_decrement <= np.sqrt(eps)) | (t == 0):
            break

        x += t * np.asarray(direction)

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")

    return x, values


def compute_average(path):
    w_average = np.zeros(path[0].shape)
    L = len(path)
    for i in range(L):
        w_average += path[i]
    return w_average / L


def two_loop_recursion(s, y, gradient):
    q = gradient.copy()
    M = len(s)
    n = s[0].shape[0]

    rho = []
    for i in range(M):
        rho.append(float(1 / s[i].T.dot(y[i])))

    coef = float(s[-1].T.dot(y[-1]) / y[-1].T.dot(y[-1]))
    H = coef * np.eye(n)

    alpha = np.zeros(M)
    for i in range(M - 1, -1, -1):
        alpha[i] = rho[i] * float(s[i].T.dot(q))
        q -= alpha[i] * y[i]

    r = H.dot(q)

    for i in range(M):
        beta = rho[i] * float(y[i].T.dot(r))
        r += s[i] * (alpha[i] - beta)

    return r


def sgd(func, initial_w, epoch, m, step_size, batch_size, n):
    w = initial_w.copy()
    w_path = [w.copy()]
    time_stamp = [0]
    time_start = time.time()

    while True:
        if len(w_path) > epoch:
            return w_path, time_stamp
        for i in range(m):
            index = np.random.choice(n, batch_size, replace=False)
            sto_gradient = func(w, 3, index)
            direction = -np.asarray(sto_gradient)
            w += step_size * direction
        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)


def sgd_bb(func, initial_w, epoch, m, batch_size, n):
    w = initial_w.copy()
    w_tilde = w.copy()
    gradient = np.zeros(w.shape)
    beta = 10 / m

    w_path = [w.copy()]

    time_stamp = [0]
    time_start = time.time()

    while True:
        if len(w_path) == 1:
            step_size = 0.01
        else:
            st = w_tilde - w_before
            yt = gradient - gradient_before
            step_size = float(np.linalg.norm(st, 2) ** 2 / abs(st.T.dot(yt))) / m
        gradient_before = gradient.copy()
        w_before = w_tilde.copy()

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, batch_size, replace=False)
            sto_gradient = func(w, 3, index)
            direction = -sto_gradient
            w += step_size * direction
            gradient = beta * sto_gradient + (1 - beta) * gradient

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def svrg(func, initial_w, epoch, m, step_size, batch_size, n):
    w = initial_w.copy()
    w_path = [w.copy()]
    w_tilde = w.copy()
    time_stamp = [0]
    time_start = time.time()

    while True:
        gradient = func(w_tilde, 1, None)
        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, batch_size, replace=False)
            sto_gradient = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            direction = -np.asarray(sto_gradient - sto_gradient_tilde + gradient)
            w += step_size * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def svrg_bb(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_tilde = w.copy()

    w_path = [w.copy()]

    time_stamp = [0]
    time_start = time.time()

    while True:
        gradient = func(w_tilde, 1, None)
        if len(w_path) == 1:
            step_size = 0.1
        else:
            st = w_tilde - w_before
            yt = gradient - gradient_before
            step_size = float(np.linalg.norm(st, 2) ** 2 / (m * st.T.dot(yt)))
        gradient_before = gradient.copy()
        w_before = w_tilde.copy()

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            direction = -(sto_gradient - sto_gradient_tilde + gradient)
            w += step_size * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def slbfgs(func, initial_w, epoch, m, M, L, step_size, batch_size, n):
    w = initial_w.copy()
    w_tilde = w.copy()

    w_path = [w.copy()]
    path = [w.copy()]
    w_average = []
    time_stamp = [0]
    time_start = time.time()
    hessian_batch = 10 * batch_size

    t = -1
    s = []
    y = []

    while True:
        gradient = func(w_tilde, 1, None)

        for i in range(m):

            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, batch_size, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)

            if len(s) < 1:
                direction = -sto_gradient
            else:
                direction = -two_loop_recursion(s, y, sto_gradient)

            w += step_size * direction
            path.append(w.copy())

            if (len(path) - 1) % L == 0:
                t += 1
                average = compute_average(path)
                path = []
                w_average.append(average.copy())
                if len(w_average) > 2:
                    w_average.pop(0)

                if t > 0:
                    st = w_average[1] - w_average[0]

                    # index = np.random.choice(n, batch_size, replace=False)
                    # yt = func(w_average[1], 3, index) - func(w_average[0], 3, index)

                    index = np.random.choice(n, hessian_batch, replace=False)
                    sto_hess = func(w_average[1], 4, index)
                    yt = sto_hess.dot(st)

                    if float(yt.T.dot(st)) > 1e-30:
                        s.append(st.copy())
                        y.append(yt.copy())
                        if len(s) > M:
                            s.pop(0)
                            y.pop(0)

        w_tilde = w.copy()
        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)


def ssm(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_path = [w.copy()]
    w_tilde = w.copy()
    time_stamp = [0]
    time_start = time.time()

    while True:
        gradient = func(w_tilde, 1, None)
        difference = func(w_tilde + gradient, 1, None) - gradient
        eta = (b / np.sqrt(m)) * (np.linalg.norm(gradient, 2) ** 2 / float(gradient.T.dot(difference)))

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            w += eta * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def ssm_sto(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_path = [w.copy()]
    w_tilde = w.copy()
    time_stamp = [0]
    time_start = time.time()

    while True:
        gradient = func(w_tilde, 1, None)

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)

            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            difference = func(w + sto_gradient, 3, index) - sto_gradient_w
            eta = (b / np.sqrt(m)) * (np.linalg.norm(sto_gradient, 2) ** 2 / float(sto_gradient.T.dot(difference)))

            w += eta * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def ssbb(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_tilde = w.copy()
    w_path = [w.copy()]
    time_stamp = [0]
    time_start = time.time()
    t = 1

    while True:
        gradient = func(w_tilde, 1, None)
        if t == 1:
            beta = -0.1
        else:
            s = w_tilde - w_tilde_before
            y = gradient - gradient_before
            beta = - float(s.T.dot(s)) / float(s.T.dot(y))
        w_tilde_before = w_tilde.copy()
        gradient_before = gradient.copy()
        t += 1

        difference = func(w + beta * gradient, 1, None) - gradient
        eta = (1 / (np.sqrt(m))) * beta * float(gradient.T.dot(gradient)) / float(gradient.T.dot(difference))
        # print(eta)

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            w += eta * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def ssbb_sto(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_tilde = w.copy()
    w_path = [w.copy()]
    time_stamp = [0]
    time_start = time.time()
    t = 1

    while True:
        gradient = func(w_tilde, 1, None)

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            if t == 1:
                beta = -0.1
            else:
                s = w - w_before
                y = sto_gradient - gradient_before
                beta = -float(s.T.dot(s)) / float(s.T.dot(y))
            w_before = w.copy()
            gradient_before = sto_gradient.copy()

            difference = func(w + beta * sto_gradient, 3, index) - sto_gradient_w
            eta = (b / np.sqrt(m)) * (beta * np.linalg.norm(sto_gradient, 2) ** 2) / float(sto_gradient.T.dot(difference))

            w += eta * direction
            t += 1

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def quasi_ssm(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_path = [w.copy()]
    w_tilde = w.copy()
    time_stamp = [0]
    time_start = time.time()

    while True:
        gradient = func(w_tilde, 1, None)
        difference = func(w_tilde + gradient, 1, None) - gradient
        eta = (b / np.sqrt(m)) * (float(gradient.T.dot(difference)) / np.linalg.norm(difference, 2) ** 2)

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            w += eta * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def quasi_ssbb(func, initial_w, epoch, m, b, n):
    w = initial_w.copy()
    w_tilde = w.copy()
    w_path = [w.copy()]
    time_stamp = [0]
    time_start = time.time()
    t = 1

    while True:
        gradient = func(w_tilde, 1, None)
        if t == 1:
            beta = -0.1
        else:
            s = w_tilde - w_tilde_before
            y = gradient - gradient_before
            beta = - float(s.T.dot(s)) / float(s.T.dot(y))
        w_tilde_before = w_tilde.copy()
        gradient_before = gradient.copy()
        t += 1

        difference = func(w + beta * gradient, 1, None) - gradient
        eta = (b / np.sqrt(m)) * beta * (float(gradient.T.dot(difference)) / float(difference.T.dot(difference)))
        # print(eta)

        for i in range(m):
            if len(w_path) > epoch:
                return w_path, time_stamp

            index = np.random.choice(n, b, replace=False)
            sto_gradient_w = func(w, 3, index)
            sto_gradient_tilde = func(w_tilde, 3, index)
            sto_gradient = np.asarray(sto_gradient_w - sto_gradient_tilde + gradient)
            direction = -sto_gradient

            w += eta * direction

        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
        w_tilde = w.copy()


def adam(func, initial_w, epoch, m, batch_size, n):
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    w = initial_w.copy()
    w_path = [w.copy()]
    time_stamp = [0]
    time_start = time.time()
    t = 1
    v = np.zeros(np.shape(w))
    u = np.zeros(np.shape(w))
    alpha = 0.005

    while True:
        if len(w_path) > epoch:
            return w_path, time_stamp
        for i in range(m):
            index = np.random.choice(n, batch_size, replace=False)
            sto_gradient = func(w, 3, index)
            g = np.asarray(sto_gradient)
            u = beta_1 * u + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * (np.multiply(g, g))
            u_hat = u / (1 - pow(beta_1, t))
            v_hat = v / (1 - pow(beta_2, t))

            w -= alpha * np.divide(u_hat, np.sqrt(v_hat) + epsilon)
            # w += step_size / np.sqrt(t) * direction
            t += 1
        w_path.append(w.copy())
        time_end = time.time()
        time_stamp.append(time_end - time_start)
