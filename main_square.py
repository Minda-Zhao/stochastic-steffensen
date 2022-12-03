import numpy as np
import matplotlib.pyplot as plt
import square as sq
import algorithms as alg
from sklearn.datasets import load_svmlight_file

np.random.seed(0)

# synthetic data
n = 10000
d = 100
beta_star = np.random.randn(d)
X_train = np.random.randn(n, d)
y_train = X_train.dot(beta_star) + np.random.normal(0, 1, size=n)
y_train = np.asmatrix(y_train).T
print(X_train.shape, y_train.shape)

# scaling
X_train = np.divide(X_train, np.max(abs(X_train), 0))
y_train = y_train / max(abs(y_train))

# initialization
repeat = 10
n = X_train.shape[0]
d = X_train.shape[1]
Lambda = 1e-5
M = 10
L = 10
func_sto = lambda w, order, index: sq.square(y=y_train, X=X_train, w=w, order=order, index=index, Lambda=Lambda)

# Use Newton method to get the optimal solution
func = lambda w, order: sq.square(y_train, X_train, w, order, None, Lambda=Lambda)
initial_w = np.asmatrix(np.zeros(shape=(d, 1)))
eps = 1e-32
maximum_iteration = 65536
alpha = 0.4
beta = 0.9
w_opt, values = alg.newton(func, initial_w, eps, maximum_iteration, alg.backtracking, alpha, beta)
value_opt = min(values)
print(value_opt, np.linalg.norm(w_opt, 2))

# Choose Task
# mode = 'batch'
# mode = 'stochastic stepsize'
# mode = 'compare SS'
mode = 'compare'

if mode == 'batch':
    epoch = 20
    batch_size = [1, 2, 4, 8, 16]
    markers = ['o', 'v', 'p', 's', '+', '*']
    colors = ['b', 'r', 'm', 'c', 'y', 'k']
    m = int(n / 4)
    for k in range(len(batch_size)):
        b = batch_size[k]
        x_axis = np.linspace(0, (epoch + 1) * (2 + 2 * m * b / n), epoch + 1)
        error_ssbb = np.zeros((epoch + 1, repeat))
        time_stamps_ssbb = np.zeros((epoch + 1, repeat))
        for i in range(repeat):
            w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, b, n)
            for j in range(len(w_path_ssbb)):
                error = func(w_path_ssbb[j], 0) - value_opt
                error_ssbb[j, i] = error
                time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
        error_ssbb = np.mean(error_ssbb, axis=1)
        plt.semilogy(x_axis, error_ssbb, linewidth=2, label="batch size = " + str(b), marker=markers[k],
                     color=colors[k])
    plt.xlabel('passes through data')
    plt.ylabel('log(suboptimality)')
    plt.xlim((0, 50))
    # plt.ylim((1e-5, 1))
    plt.legend()
    plt.savefig('./plot/ridge/ridge_passes_batch.png')

elif mode == 'stochastic stepsize':
    # initialization
    epoch = 10
    m = int(2 * n)
    b = 4
    x_axis = np.linspace(0, (epoch + 1) * (2 + 2 * m * b / n), epoch + 1)
    x_axis_sto = np.linspace(0, (epoch + 1) * (1 + 3 * m * b / n), epoch + 1)

    # SSM
    error_ssm = np.zeros((epoch + 1, repeat))
    time_stamps_ssm = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssm, time_stamp_ssm = alg.ssm(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssm)):
            error = func(w_path_ssm[j], 0) - value_opt
            error_ssm[j, i] = error
            time_stamps_ssm[j, i] = time_stamp_ssm[j]
    error_ssm = np.mean(error_ssm, axis=1)
    time_stamps_ssm = np.mean(time_stamps_ssm, axis=1)

    # qSSM
    error_qssm = np.zeros((epoch + 1, repeat))
    time_stamps_qssm = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_qssm, time_stamp_qssm = alg.quasi_ssm(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_qssm)):
            error = func(w_path_qssm[j], 0) - value_opt
            error_qssm[j, i] = error
            time_stamps_qssm[j, i] = time_stamp_qssm[j]
    error_qssm = np.mean(error_qssm, axis=1)
    time_stamps_qssm = np.mean(time_stamps_qssm, axis=1)

    # SSM with stochastic stepsize
    error_ssm_sto = np.zeros((epoch + 1, repeat))
    time_stamps_ssm_sto = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssm_sto, time_stamp_ssm_sto = alg.ssm_sto(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssm_sto)):
            error = func(w_path_ssm_sto[j], 0) - value_opt
            error_ssm_sto[j, i] = error
            time_stamps_ssm_sto[j, i] = time_stamp_ssm_sto[j]
    error_ssm_sto = np.mean(error_ssm_sto, axis=1)
    time_stamps_ssm_sto = np.mean(time_stamps_ssm_sto, axis=1)

    # SSBB
    error_ssbb = np.zeros((epoch + 1, repeat))
    time_stamps_ssbb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssbb)):
            error = func(w_path_ssbb[j], 0) - value_opt
            error_ssbb[j, i] = error
            time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
    error_ssbb = np.mean(error_ssbb, axis=1)
    time_stamps_ssbb = np.mean(time_stamps_ssbb, axis=1)

    # qSSBB
    error_qssbb = np.zeros((epoch + 1, repeat))
    time_stamps_qssbb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_qssbb, time_stamp_qssbb = alg.quasi_ssbb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_qssbb)):
            error = func(w_path_qssbb[j], 0) - value_opt
            error_qssbb[j, i] = error
            time_stamps_qssbb[j, i] = time_stamp_qssbb[j]
    error_qssbb = np.mean(error_qssbb, axis=1)
    time_stamps_qssbb = np.mean(time_stamps_qssbb, axis=1)

    # SSBB with stochastic stepsize
    error_ssbb_sto = np.zeros((epoch + 1, repeat))
    time_stamps_ssbb_sto = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssbb_sto, time_stamp_ssbb_sto = alg.ssbb_sto(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssbb_sto)):
            error = func(w_path_ssbb_sto[j], 0) - value_opt
            error_ssbb_sto[j, i] = error
            time_stamps_ssbb_sto[j, i] = time_stamp_ssbb_sto[j]
    error_ssbb_sto = np.mean(error_ssbb_sto, axis=1)
    time_stamps_ssbb_sto = np.mean(time_stamps_ssbb_sto, axis=1)

    # normalized error vs passes
    plt.semilogy(x_axis, error_ssm, linewidth=2, label='SSM', marker='s', color='b')
    plt.semilogy(x_axis, error_qssm, linewidth=2, label='quasi-SSM', marker='p', color='g')
    plt.semilogy(x_axis_sto, error_ssm_sto, linewidth=2, label='SSM-Sto', marker='*', color='r')
    plt.semilogy(x_axis, error_ssbb, linewidth=2, label='SSBB', marker='h', color='m')
    plt.semilogy(x_axis, error_qssbb, linewidth=2, label='quasi-SSBB', marker='+', color='y')
    plt.semilogy(x_axis_sto, error_ssbb_sto, linewidth=2, label='SSBB-Sto', marker='*', color='k')
    plt.xlabel('passes through data')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 200))
    plt.savefig('./plot/ridge/ridge_sto_passes.png')
    plt.close()

elif mode == 'compare SS':
    # initialization
    epoch = 15
    m = int(2 * n)
    b = 4
    x_axis = np.linspace(0, (epoch + 1) * (2 + 2 * m * b / n), epoch + 1)

    # SSM
    error_ssm = np.zeros((epoch + 1, repeat))
    time_stamps_ssm = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssm, time_stamp_ssm = alg.ssm(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssm)):
            error = func(w_path_ssm[j], 0) - value_opt
            error_ssm[j, i] = error
            time_stamps_ssm[j, i] = time_stamp_ssm[j]
    error_ssm = np.mean(error_ssm, axis=1)
    time_stamps_ssm = np.mean(time_stamps_ssm, axis=1)

    # qSSM
    error_qssm = np.zeros((epoch + 1, repeat))
    time_stamps_qssm = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_qssm, time_stamp_qssm = alg.quasi_ssm(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_qssm)):
            error = func(w_path_qssm[j], 0) - value_opt
            error_qssm[j, i] = error
            time_stamps_qssm[j, i] = time_stamp_qssm[j]
    error_qssm = np.mean(error_qssm, axis=1)
    time_stamps_qssm = np.mean(time_stamps_qssm, axis=1)

    # SSBB
    error_ssbb = np.zeros((epoch + 1, repeat))
    time_stamps_ssbb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssbb)):
            error = func(w_path_ssbb[j], 0) - value_opt
            error_ssbb[j, i] = error
            time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
    error_ssbb = np.mean(error_ssbb, axis=1)
    time_stamps_ssbb = np.mean(time_stamps_ssbb, axis=1)

    # qSSBB
    error_qssbb = np.zeros((epoch + 1, repeat))
    time_stamps_qssbb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_qssbb, time_stamp_qssbb = alg.quasi_ssbb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_qssbb)):
            error = func(w_path_qssbb[j], 0) - value_opt
            error_qssbb[j, i] = error
            time_stamps_qssbb[j, i] = time_stamp_qssbb[j]
    error_qssbb = np.mean(error_qssbb, axis=1)
    time_stamps_qssbb = np.mean(time_stamps_qssbb, axis=1)

    # normalized error vs passes
    plt.semilogy(x_axis, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
    plt.semilogy(x_axis, error_qssm, linewidth=2, label='quasi-SSM', marker='v', color='r')
    plt.semilogy(x_axis, error_ssbb, linewidth=2, label='SSBB', marker='s', color='c')
    plt.semilogy(x_axis, error_qssbb, linewidth=2, label='quasi-SSBB', marker='+', color='y')
    plt.xlabel('passes through data')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 160))
    plt.savefig('./plot/ridge/ridge_quasi_passes.png')
    plt.close()

    # normalized error vs time
    plt.semilogy(time_stamps_ssm, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
    plt.semilogy(time_stamps_qssm, error_qssm, linewidth=2, label='quasi-SSM', marker='v', color='r')
    plt.semilogy(time_stamps_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='s', color='c')
    plt.semilogy(time_stamps_qssbb, error_qssbb, linewidth=2, label='quasi-SSBB', marker='+', color='y')
    plt.xlabel('running time (s)')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 17.5))
    # plt.ylim((1e-19, 1))
    plt.savefig('./plot/ridge/ridge_quasi_time.png')
    plt.close()

elif mode == 'compare':
    # initialization
    epoch = 10
    # m = int(n / 32)
    m = 4 * n
    b = 4
    x_axis_sgd = np.linspace(0, (2 * epoch + 1) * (m * b / n), 2 * epoch + 1)
    x_axis_svrg = np.linspace(0, (epoch + 1) * (1 + 2 * m * b / n), epoch + 1)
    x_axis_slbfgs = np.linspace(0, (epoch + 1) * (1 + 2 * m * b / n), epoch + 1)
    x_axis_ssbb = np.linspace(0, (epoch + 1) * (2 + 2 * m * b / n), epoch + 1)

    # # Grid Search for step size of SGD, the best is 0.01
    # eta_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    # for eta in eta_list:
    #     error_sgd = np.zeros((2 * epoch + 1, repeat))
    #     time_stamps_sgd = np.zeros((2 * epoch + 1, repeat))
    #     for i in range(repeat):
    #         w_path_sgd, time_stamp_sgd = alg.sgd(func_sto, initial_w, 2 * epoch, m, eta, b, n)
    #         for j in range(len(w_path_sgd)):
    #             error = func(w_path_sgd[j], 0) - value_opt
    #             error_sgd[j, i] = error
    #             time_stamps_sgd[j, i] = time_stamp_sgd[j]
    #     error_sgd = np.mean(error_sgd, axis=1)
    #     plt.semilogy(x_axis_sgd, error_sgd, linewidth=2, label="eta = "+str(eta))
    # plt.xlabel('passes through data')
    # plt.ylabel('log(suboptimality)')
    # plt.legend()
    # plt.show()

    # # Grid Search for step size of SVRG, the best is 0.01
    # eta_list = [0.1, 0.01, 0.001]
    # for eta in eta_list:
    #     error_svrg = np.zeros((epoch + 1, repeat))
    #     time_stamps_svrg = np.zeros((epoch + 1, repeat))
    #     for i in range(repeat):
    #         w_path_svrg, time_stamp_svrg = alg.svrg(func_sto, initial_w, epoch, m, eta, b, n)
    #         for j in range(len(w_path_svrg)):
    #             error = func(w_path_svrg[j], 0) - value_opt
    #             error_svrg[j, i] = error
    #             time_stamps_svrg[j, i] = time_stamp_svrg[j]
    #     error_svrg = np.mean(error_svrg, axis=1)
    #     plt.semilogy(x_axis_svrg, error_svrg, linewidth=2, label="eta = "+str(eta))
    # plt.xlabel('passes through data')
    # plt.ylabel('log(suboptimality)')
    # plt.legend()
    # plt.show()

    # # Grid Search for step size of SLBFGS, the best is 0.01
    # eta_list = [0.01, 0.005, 0.001]
    # for eta in eta_list:
    #     error_slbfgs = np.zeros((epoch + 1, repeat))
    #     time_stamps_slbfgs = np.zeros((epoch + 1, repeat))
    #     for i in range(repeat):
    #         w_path_slbfgs, time_stamp_slbfgs = alg.slbfgs(func_sto, initial_w, epoch, m, M, L, eta, b, n)
    #         for j in range(len(w_path_slbfgs)):
    #             error = func(w_path_slbfgs[j], 0) - value_opt
    #             error_slbfgs[j, i] = error
    #             time_stamps_slbfgs[j, i] = time_stamp_slbfgs[j]
    #     error_slbfgs = np.mean(error_slbfgs, axis=1)
    #     plt.semilogy(x_axis_slbfgs, error_slbfgs, linewidth=2, label="eta = "+str(eta))
    # plt.xlabel('passes through data')
    # plt.ylabel('log(suboptimality)')
    # plt.legend()
    # plt.show()

    # plots
    # sgd
    error_sgd = np.zeros((2 * epoch + 1, repeat))
    time_stamps_sgd = np.zeros((2 * epoch + 1, repeat))
    for i in range(repeat):
        w_path_sgd, time_stamp_sgd = alg.sgd(func_sto, initial_w, 2 * epoch, m, 0.01, b, n)
        for j in range(len(w_path_sgd)):
            error = func(w_path_sgd[j], 0) - value_opt
            error_sgd[j, i] = error
            time_stamps_sgd[j, i] = time_stamp_sgd[j]
    error_sgd = np.mean(error_sgd, axis=1)
    time_stamps_sgd = np.mean(time_stamps_sgd, axis=1)

    # svrg
    error_svrg = np.zeros((epoch + 1, repeat))
    time_stamps_svrg = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_svrg, time_stamp_svrg = alg.svrg(func_sto, initial_w, epoch, m, 0.05, b, n)
        for j in range(len(w_path_svrg)):
            error = func(w_path_svrg[j], 0) - value_opt
            error_svrg[j, i] = error
            time_stamps_svrg[j, i] = time_stamp_svrg[j]
    error_svrg = np.mean(error_svrg, axis=1)
    time_stamps_svrg = np.mean(time_stamps_svrg, axis=1)

    # svrg-bb
    error_svrg_bb = np.zeros((epoch + 1, repeat))
    time_stamps_svrg_bb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_svrg_bb, time_stamp_svrg_bb = alg.svrg_bb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_svrg_bb)):
            error = func(w_path_svrg_bb[j], 0) - value_opt
            error_svrg_bb[j, i] = error
            time_stamps_svrg_bb[j, i] = time_stamp_svrg_bb[j]
    error_svrg_bb = np.mean(error_svrg_bb, axis=1)
    time_stamps_svrg_bb = np.mean(time_stamps_svrg_bb, axis=1)

    # slbfgs
    error_slbfgs = np.zeros((epoch + 1, repeat))
    time_stamps_slbfgs = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_slbfgs, time_stamp_slbfgs = alg.slbfgs(func_sto, initial_w, epoch, m, M, L, 0.01, b, n)
        for j in range(len(w_path_slbfgs)):
            error = func(w_path_slbfgs[j], 0) - value_opt
            error_slbfgs[j, i] = error
            time_stamps_slbfgs[j, i] = time_stamp_slbfgs[j]
    error_slbfgs = np.mean(error_slbfgs, axis=1)
    time_stamps_slbfgs = np.mean(time_stamps_slbfgs, axis=1)

    # ssbb
    error_ssbb = np.zeros((epoch + 1, repeat))
    time_stamps_ssbb = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssbb)):
            error = func(w_path_ssbb[j], 0) - value_opt
            error_ssbb[j, i] = error
            time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
    error_ssbb = np.mean(error_ssbb, axis=1)
    time_stamps_ssbb = np.mean(time_stamps_ssbb, axis=1)

    # normalized error vs passes
    plt.semilogy(x_axis_sgd, error_sgd, linewidth=2, label='SGD', marker='o', color='b')
    plt.semilogy(x_axis_svrg, error_svrg, linewidth=2, label='SVRG', marker='v', color='r')
    plt.semilogy(x_axis_svrg, error_svrg_bb, linewidth=2, label='SVRG-BB', marker='s', color='c')
    plt.semilogy(x_axis_slbfgs, error_slbfgs, linewidth=2, label='SLBFGS', marker='+', color='y')
    plt.semilogy(x_axis_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='D', color='k')
    plt.xlabel('passes through data')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 300))
    plt.ylim((1e-10, 1))
    plt.savefig('./plot/ridge/ridge_passes.png')
    plt.close()

    # normalized error vs time
    plt.semilogy(time_stamps_sgd, error_sgd, linewidth=2, label='SGD', marker='o', color='b')
    plt.semilogy(time_stamps_svrg, error_svrg, linewidth=2, label='SVRG', marker='v', color='r')
    plt.semilogy(time_stamps_svrg_bb, error_svrg_bb, linewidth=2, label='SVRG-BB', marker='s', color='c')
    plt.semilogy(time_stamps_slbfgs, error_slbfgs, linewidth=2, label='SLBFGS', marker='+', color='y')
    plt.semilogy(time_stamps_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='D', color='k')
    plt.xlabel('running time (s)')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 70))
    plt.ylim((1e-10, 1))
    plt.savefig('./plot/ridge/ridge_time.png')
    plt.close()
