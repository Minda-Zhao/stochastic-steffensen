import numpy as np
import matplotlib.pyplot as plt
import logistic as lg
import algorithms as alg
from sklearn.datasets import load_svmlight_file
import cvxpy as cp

# np.random.seed(0)

data = load_svmlight_file("dataset/w6a.txt")
X_train = data[0].todense()
y_train = data[1]

# Change label to 1 and -1
y_train = (y_train == 1).astype(int)
y_train[y_train == 0] = -1
print(X_train.shape, y_train.shape)

# initialization
repeat = 10
n = X_train.shape[0]
d = X_train.shape[1]
Lambda = 1e-4
M = 10
L = 10
func_sto = lambda w, order, index: lg.logistic(y=y_train, X=X_train, w=w, order=order, index=index, Lambda=Lambda)
func = lambda w, order: lg.logistic(y_train, X_train, w, order, None, Lambda=Lambda)
initial_w = np.asmatrix(np.zeros(shape=(d, 1)))

# # Use cvxpy to get the optimal solution
# w = cp.Variable(d)
# Xw = X_train @ w
# yXw = cp.multiply(y_train, Xw)
# prob = cp.Problem(cp.Minimize(cp.sum(cp.logistic(-yXw)) / n + Lambda * cp.norm(w, 2) ** 2 / 2))
# prob.solve()
# value_opt = prob.value
# print(value_opt, np.linalg.norm(w.value, 2))

# Use Newton method to get the optimal solution
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
mode = 'compare'
# mode = 'compare SS'

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
    # plt.xlim((0, 50))
    # plt.ylim((1e-5, 1))
    plt.legend()
    plt.savefig('./plot/logistic/logistic_passes_batch.png')

elif mode == 'stochastic stepsize':
    # initialization
    epoch = 20
    m = int(n / 16)
    b = 8
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

    # SSM with different coefficient
    error_ssm_coef = np.zeros((epoch + 1, repeat))
    time_stamps_ssm_coef = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssm_coef, time_stamp_ssm_coef = alg.ssm_coef(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssm_coef)):
            error = func(w_path_ssm_coef[j], 0) - value_opt
            error_ssm_coef[j, i] = error
            time_stamps_ssm_coef[j, i] = time_stamp_ssm_coef[j]
    error_ssm_coef = np.mean(error_ssm_coef, axis=1)
    time_stamps_ssm_coef = np.mean(time_stamps_ssm_coef, axis=1)

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

    # SSBB with different coefficient
    error_ssbb_coef = np.zeros((epoch + 1, repeat))
    time_stamps_ssbb_coef = np.zeros((epoch + 1, repeat))
    for i in range(repeat):
        w_path_ssbb_coef, time_stamp_ssbb_coef = alg.ssbb_coef(func_sto, initial_w, epoch, m, b, n)
        for j in range(len(w_path_ssbb_coef)):
            error = func(w_path_ssbb_coef[j], 0) - value_opt
            error_ssbb_coef[j, i] = error
            time_stamps_ssbb_coef[j, i] = time_stamp_ssbb_coef[j]
    error_ssbb_coef = np.mean(error_ssbb_coef, axis=1)
    time_stamps_ssbb_coef = np.mean(time_stamps_ssbb_coef, axis=1)

    # # SVRG-BB
    # error_svrgbb = np.zeros((epoch_svrg + 1, repeat))
    # time_stamps_svrgbb = np.zeros((epoch_svrg + 1, repeat))
    # for i in range(repeat):
    #     w_path_svrgbb, time_stamp_svrgbb = alg.svrg_bb(func_sto, initial_w, epoch_svrg, m, b, n)
    #     for j in range(len(w_path_svrgbb)):
    #         error = func(w_path_svrgbb[j], 0) - value_opt
    #         error_svrgbb[j, i] = error
    #         time_stamps_svrgbb[j, i] = time_stamp_svrgbb[j]
    # error_svrgbb = np.mean(error_svrgbb, axis=1)
    # time_stamps_svrgbb = np.mean(time_stamps_svrgbb, axis=1)

    # normalized error vs passes
    plt.semilogy(x_axis, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
    plt.semilogy(x_axis_sto, error_ssm_sto, linewidth=2, label='SSM-Sto', marker='v', color='r')
    plt.semilogy(x_axis, error_ssm_coef, linewidth=2, label=r'SSM-$\frac{1}{\sqrt{m}}$', marker='p', color='m')
    plt.semilogy(x_axis, error_ssbb, linewidth=2, label='SSBB', marker='s', color='c')
    plt.semilogy(x_axis_sto, error_ssbb_sto, linewidth=2, label='SSBB-Sto', marker='+', color='y')
    plt.semilogy(x_axis, error_ssbb_coef, linewidth=2, label=r'SSBB-$\frac{1}{\sqrt{m}}$', marker='D', color='k')
    plt.xlabel('passes through data')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 50))
    plt.savefig('./plot/logistic/logistic_sto_passes.png')
    plt.close()

    # normalized error vs time
    plt.semilogy(time_stamps_ssm, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
    plt.semilogy(time_stamps_ssm_sto, error_ssm_sto, linewidth=2, label='SSM-Sto', marker='v', color='r')
    plt.semilogy(time_stamps_ssm_coef, error_ssm_coef, linewidth=2, label=r'SSM-$\frac{1}{\sqrt{m}}$', marker='p',
                 color='m')
    plt.semilogy(time_stamps_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='s', color='c')
    plt.semilogy(time_stamps_ssbb_sto, error_ssbb_sto, linewidth=2, label='SSBB-Sto', marker='+', color='y')
    plt.semilogy(time_stamps_ssbb_coef, error_ssbb_coef, linewidth=2, label=r'SSBB-$\frac{1}{\sqrt{m}}$', marker='D',
                 color='k')
    plt.xlabel('running time (s)')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    plt.xlim((0, 6))
    # plt.ylim((1e-19, 1))
    plt.savefig('./plot/logistic/logistic_sto_time.png')
    plt.close()

elif mode == 'compare SS':
    # initialization
    epoch = 20
    m = int(n / 16)
    b = 16
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

    # SSBB with stochastic stepsize
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
    # plt.xlim((0, 50))
    plt.savefig('./plot/logistic/logistic_quasi_passes.png')
    plt.close()

    # normalized error vs time
    plt.semilogy(time_stamps_ssm, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
    plt.semilogy(time_stamps_qssm, error_qssm, linewidth=2, label='quasi-SSM', marker='v', color='r')
    plt.semilogy(time_stamps_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='s', color='c')
    plt.semilogy(time_stamps_qssbb, error_qssbb, linewidth=2, label='quasi-SSBB', marker='+', color='y')
    plt.xlabel('running time (s)')
    plt.ylabel('log(suboptimality)')
    plt.legend()
    # plt.xlim((0, 6))
    # plt.ylim((1e-19, 1))
    plt.savefig('./plot/logistic/logistic_quasi_time.png')
    plt.close()

elif mode == 'compare':
    # initialization
    epoch = 10
    # m = int(n / 32)
    m = 2 * n
    b = 16
    x_axis_sgd = np.linspace(0, (2 * epoch + 1) * (m * b / n), 2 * epoch + 1)
    x_axis_svrg = np.linspace(0, (epoch + 1) * (1 + 2 * m * b / n), epoch + 1)
    x_axis_slbfgs = np.linspace(0, (epoch + 1) * (1 + 2 * m * b / n), epoch + 1)
    x_axis_ssbb = np.linspace(0, (epoch + 1) * (2 + 2 * m * b / n), epoch + 1)

    # # Grid Search for step size of SGD, the best is 0.1
    # eta_list = [1, 0.5, 0.1, 0.05, 0.01]
    # for eta in eta_list:
    #     error_sgd = np.zeros((epoch + 1, repeat))
    #     time_stamps_sgd = np.zeros((epoch + 1, repeat))
    #     for i in range(repeat):
    #         w_path_sgd, time_stamp_sgd = alg.sgd(func_sto, initial_w, epoch, m, eta, b, n)
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

    # # Grid Search for step size of SVRG, the best is 1
    # eta_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
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

    # # Grid Search for step size of SLBFGS, the best is 0.001
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
        w_path_sgd, time_stamp_sgd = alg.sgd(func_sto, initial_w, 2 * epoch, m, 0.1, b, n)
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
        w_path_svrg, time_stamp_svrg = alg.svrg(func_sto, initial_w, epoch, m, 0.1, b, n)
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
        w_path_slbfgs, time_stamp_slbfgs = alg.slbfgs(func_sto, initial_w, epoch, m, M, L, 0.001, b, n)
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
    plt.xlim((0, 600))
    plt.ylim((1e-9, 1))
    plt.savefig('./plot/logistic/logistic_passes.png')
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
    plt.xlim((0, 100))
    plt.ylim((1e-9, 1))
    plt.savefig('./plot/logistic/logistic_time.png')
    plt.close()

### Compare SSM, qSSM, SSBB, qSSBB -------------------------------------------------------------------------------------

# # Grid Search for step size of SSM, the best is 0.01
# eta_list = [0.05, 0.01, 0.005, 0.001]
# for eta in eta_list:
#     error_ssm = np.zeros((epoch_steffensen + 1, repeat))
#     time_stamps_ssm = np.zeros((epoch_steffensen + 1, repeat))
#     for i in range(repeat):
#         w_path_ssm, time_stamp_ssm = alg.ssm(func_sto, initial_w, epoch_steffensen, m, b, n, eta)
#         for j in range(len(w_path_ssm)):
#             error = func(w_path_ssm[j], 0) - values_opt
#             error_ssm[j, i] = error
#             time_stamps_ssm[j, i] = time_stamp_ssm[j]
#     error_ssm = np.mean(error_ssm, axis=1)
#     plt.semilogy(x_axis_steffensen, error_ssm, linewidth=2, label="eta = "+str(eta))
# plt.xlabel('passes through data')
# plt.ylabel('log(suboptimality)')
# plt.legend()
# plt.show()

# # Grid Search for step size of qSSM, the best is 0.01
# eta_list = [0.05, 0.01, 0.005, 0.001]
# for eta in eta_list:
#     error_qssm = np.zeros((epoch_steffensen + 1, repeat))
#     time_stamps_qssm = np.zeros((epoch_steffensen + 1, repeat))
#     for i in range(repeat):
#         w_path_qssm, time_stamp_qssm = alg.quasi_ssm(func_sto, initial_w, epoch_steffensen, m, b, n, eta)
#         for j in range(len(w_path_qssm)):
#             error = func(w_path_qssm[j], 0) - values_opt
#             error_qssm[j, i] = error
#             time_stamps_qssm[j, i] = time_stamp_qssm[j]
#     error_qssm = np.mean(error_qssm, axis=1)
#     plt.semilogy(x_axis_steffensen, error_qssm, linewidth=2, label="eta = "+str(eta))
# plt.xlabel('passes through data')
# plt.ylabel('log(suboptimality)')
# plt.legend()
# plt.show()

# # Grid Search for step size of qSSBB, the best is 0.01
# eta_list = [0.1, 0.05, 0.01, 0.005, 0.001]
# for eta in eta_list:
#     error_qssbb = np.zeros((epoch_steffensen + 1, repeat))
#     time_stamps_qssbb = np.zeros((epoch_steffensen + 1, repeat))
#     for i in range(repeat):
#         w_path_qssbb, time_stamp_qssbb = alg.quasi_ssbb(func_sto, initial_w, epoch_steffensen, m, b, n, eta)
#         for j in range(len(w_path_qssbb)):
#             error = func(w_path_qssbb[j], 0) - values_opt
#             error_qssbb[j, i] = error
#             time_stamps_qssbb[j, i] = time_stamp_qssbb[j]
#     error_qssbb = np.mean(error_qssbb, axis=1)
#     plt.semilogy(x_axis_steffensen, error_qssbb, linewidth=2, label="eta = "+str(eta))
# plt.xlabel('passes through data')
# plt.ylabel('log(suboptimality)')
# plt.legend()
# plt.show()

# ## plot
# # SSM
# error_ssm = np.zeros((epoch_steffensen + 1, repeat))
# time_stamps_ssm = np.zeros((epoch_steffensen + 1, repeat))
# for i in range(repeat):
#     w_path_ssm, time_stamp_ssm = alg.ssm(func_sto, initial_w, epoch_steffensen, m, b, n, 0.01)
#     for j in range(len(w_path_ssm)):
#         error = func(w_path_ssm[j], 0) - values_opt
#         error_ssm[j, i] = error
#         time_stamps_ssm[j, i] = time_stamp_ssm[j]
# error_ssm = np.mean(error_ssm, axis=1)
# time_stamps_ssm = np.mean(time_stamps_ssm, axis=1)
#
# # qSSM
# error_qssm = np.zeros((epoch_steffensen + 1, repeat))
# time_stamps_qssm = np.zeros((epoch_steffensen + 1, repeat))
# for i in range(repeat):
#     w_path_qssm, time_stamp_qssm = alg.quasi_ssm(func_sto, initial_w, epoch_steffensen, m, b, n, 0.01)
#     for j in range(len(w_path_qssm)):
#         error = func(w_path_qssm[j], 0) - values_opt
#         error_qssm[j, i] = error
#         time_stamps_qssm[j, i] = time_stamp_qssm[j]
# error_qssm = np.mean(error_qssm, axis=1)
# time_stamps_qssm = np.mean(time_stamps_qssm, axis=1)
#
# # SSBB
# error_ssbb = np.zeros((epoch_steffensen + 1, repeat))
# time_stamps_ssbb = np.zeros((epoch_steffensen + 1, repeat))
# for i in range(repeat):
#     w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch_steffensen, m, b, n, 0.01)
#     for j in range(len(w_path_ssbb)):
#         error = func(w_path_ssbb[j], 0) - values_opt
#         error_ssbb[j, i] = error
#         time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
# error_ssbb = np.mean(error_ssbb, axis=1)
# time_stamps_ssbb = np.mean(time_stamps_ssbb, axis=1)
#
# # qSSBB
# error_qssbb = np.zeros((epoch_steffensen + 1, repeat))
# time_stamps_qssbb = np.zeros((epoch_steffensen + 1, repeat))
# for i in range(repeat):
#     w_path_qssbb, time_stamp_qssbb = alg.quasi_ssbb(func_sto, initial_w, epoch_steffensen, m, b, n, 0.01)
#     for j in range(len(w_path_qssbb)):
#         error = func(w_path_qssbb[j], 0) - values_opt
#         error_qssbb[j, i] = error
#         time_stamps_qssbb[j, i] = time_stamp_qssbb[j]
# error_qssbb = np.mean(error_qssbb, axis=1)
# time_stamps_qssbb = np.mean(time_stamps_qssbb, axis=1)
#
#
# # normalized error vs passes
# plt.semilogy(x_axis_steffensen, error_ssm, linewidth=2, label='SSM', marker='o', color='b')
# plt.semilogy(x_axis_steffensen, error_qssm, linewidth=2, label='qSSM', marker='v', color='r')
# plt.semilogy(x_axis_steffensen, error_ssbb, linewidth=2, label='SSBB', marker='D', color='k')
# plt.semilogy(x_axis_steffensen, error_qssbb, linewidth=2, label='qSSBB', marker='s', color='c')
# plt.xlabel('passes through data')
# plt.ylabel('log(suboptimality)')
# plt.legend()
# plt.xlim((0, 120))
# plt.ylim((1e-19, 1))
# plt.savefig('./plot/logistic/logistic_passes_compare.png')
# plt.show()
