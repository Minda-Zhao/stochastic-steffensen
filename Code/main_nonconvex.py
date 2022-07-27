import input_data
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import random
import algorithms as alg
import nonconvex as nc
from sklearn.datasets import load_svmlight_file

np.random.seed(0)

# ijcnn1 dataset
data = load_svmlight_file("dataset/ijcnn1.bz2")
X_train = data[0].todense()
y_train = np.asmatrix(data[1]).T
print(X_train.shape, y_train.shape)

# Change label to 1 and -1
y_train = (y_train == 0).astype(int)
y_train[y_train == 0] = -1

# initialization
repeat = 10
n = X_train.shape[0]
d = X_train.shape[1]
Lambda = 1e-4
M = 10
L = 10
passes = 60
epoch_sgd = int(passes)
epoch_svrg = int(passes / 3)
epoch_slbfgs = int(passes / 4)
epoch_steffensen = int(passes / 4)
x_axis_svrg = np.linspace(0, passes, epoch_svrg + 1)
x_axis_slbfgs = np.linspace(0, passes, epoch_slbfgs + 1)
x_axis_steffensen = np.linspace(0, passes, epoch_steffensen + 1)
func_sto = lambda w, order, index: nc.nonconvex(y=y_train, X=X_train, w=w, order=order, index=index, Lambda=Lambda)

# Use Newton method to get the optimal solution
func = lambda w, order: nc.nonconvex(y_train, X_train, w, order, None, Lambda=Lambda)
initial_w = np.asmatrix(np.zeros(shape=(d, 1)))
eps = 1e-32
maximum_iteration = 65536
alpha = 0.4
beta = 0.9
w_opt, values = alg.newton(func, initial_w, eps, maximum_iteration, alg.backtracking, alpha, beta)
values_opt = min(values)
print(values_opt, np.linalg.norm(w_opt, 2))

# Performance of different batches
m = 100
epoch = 15
for i in range(repeat):
    w_path_ssbb1, time_stamp_ssbb1 = alg.ssbb1(func_sto, initial_w, epoch, m, n)
    error_ssbb1 = np.zeros((epoch + 1, repeat))
    for j in range(len(w_path_ssbb1)):
        error = func(w_path_ssbb1[j], 0) - values_opt
        error_ssbb1[j, i] = error
error_ssbb1 = np.mean(error_ssbb1, axis=1)
plt.semilogy(error_ssbb1, linewidth=2, label='b = 1', marker='o', color='b')

for i in range(repeat):
    w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, 8, n)
    error_ssbb = np.zeros((epoch + 1, repeat))
    for j in range(len(w_path_ssbb)):
        error = func(w_path_ssbb[j], 0) - values_opt
        error_ssbb[j, i] = error
error_ssbb = np.mean(error_ssbb, axis=1)
plt.semilogy(error_ssbb, linewidth=2, label='b = 8', marker='^', color='g')

for i in range(repeat):
    w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, 16, n)
    error_ssbb = np.zeros((epoch + 1, repeat))
    for j in range(len(w_path_ssbb)):
        error = func(w_path_ssbb[j], 0) - values_opt
        error_ssbb[j, i] = error
error_ssbb = np.mean(error_ssbb, axis=1)
plt.semilogy(error_ssbb, linewidth=2, label='b = 16', marker='v', color='r')

# for i in range(repeat):
#     w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch, m, 32, n)
#     error_ssbb = np.zeros((epoch + 1, repeat))
#     for j in range(len(w_path_ssbb)):
#         error = func(w_path_ssbb[j], 0) - values_opt
#         error_ssbb[j, i] = error
#     error_ssbb = np.mean(error_ssbb, axis=1)
#     plt.semilogy(error_ssbb, linewidth=2, label='b = 32', marker='s', color='c')

plt.xlabel('iterations')
plt.ylabel('log(suboptimality)')
plt.legend()
plt.xlim((0, 15))
plt.ylim((1e-17, 10))
plt.savefig('./plot/nonconvex/batch.png')
plt.show()

# initialization
b = 64
m = int(n / b)

# plots
# sgd
error_sgd = np.zeros((epoch_sgd + 1, repeat))
time_stamps_sgd = np.zeros((epoch_sgd + 1, repeat))
for i in range(repeat):
    w_path_sgd, time_stamp_sgd = alg.sgd(func_sto, initial_w, epoch_sgd, m, 1, b, n)
    for j in range(len(w_path_sgd)):
        error = func(w_path_sgd[j],0) - values_opt
        error_sgd[j, i] = error
        time_stamps_sgd[j, i] = time_stamp_sgd[j]
error_sgd = np.mean(error_sgd, axis=1)
time_stamps_sgd = np.mean(time_stamps_sgd, axis=1)

# sgd-bb
error_sgd_bb = np.zeros((epoch_sgd + 1, repeat))
time_stamps_sgd_bb = np.zeros((epoch_sgd + 1, repeat))
for i in range(repeat):
    w_path_sgd_bb, time_stamp_sgd_bb = alg.sgd_bb(func_sto, initial_w, epoch_sgd, m, b, n)
    for j in range(len(w_path_sgd_bb)):
        error = func(w_path_sgd_bb[j], 0) - values_opt
        error_sgd_bb[j, i] = error
        time_stamps_sgd_bb[j, i] = time_stamp_sgd_bb[j]
error_sgd_bb = np.mean(error_sgd_bb, axis=1)
time_stamps_sgd_bb = np.mean(time_stamps_sgd_bb, axis=1)

# svrg
error_svrg = np.zeros((epoch_svrg + 1, repeat))
time_stamps_svrg = np.zeros((epoch_svrg + 1, repeat))
for i in range(repeat):
    w_path_svrg, time_stamp_svrg = alg.svrg(func_sto, initial_w, epoch_svrg, m, 1, b, n)
    for j in range(len(w_path_svrg)):
        error = func(w_path_svrg[j], 0) - values_opt
        error_svrg[j, i] = error
        time_stamps_svrg[j, i] = time_stamp_svrg[j]
error_svrg = np.mean(error_svrg, axis=1)
time_stamps_svrg = np.mean(time_stamps_svrg, axis=1)

# svrg-bb
error_svrg_bb = np.zeros((epoch_svrg + 1, repeat))
time_stamps_svrg_bb = np.zeros((epoch_svrg + 1, repeat))
for i in range(repeat):
    w_path_svrg_bb, time_stamp_svrg_bb = alg.svrg_bb(func_sto, initial_w, epoch_svrg, m, b, n)
    for j in range(len(w_path_svrg_bb)):
        error = func(w_path_svrg_bb[j], 0) - values_opt
        error_svrg_bb[j, i] = error
        time_stamps_svrg_bb[j, i] = time_stamp_svrg_bb[j]
error_svrg_bb = np.mean(error_svrg_bb, axis=1)
time_stamps_svrg_bb = np.mean(time_stamps_svrg_bb, axis=1)

# slbfgs
error_slbfgs = np.zeros((epoch_slbfgs + 1, repeat))
time_stamps_slbfgs = np.zeros((epoch_slbfgs + 1, repeat))
for i in range(repeat):
    w_path_slbfgs, time_stamp_slbfgs = alg.slbfgs(func_sto, initial_w, epoch_slbfgs, m, M, L, 0.01, b, n)
    for j in range(len(w_path_slbfgs)):
        error = func(w_path_slbfgs[j], 0) - values_opt
        error_slbfgs[j, i] = error
        time_stamps_slbfgs[j, i] = time_stamp_slbfgs[j]
error_slbfgs = np.mean(error_slbfgs, axis=1)
time_stamps_slbfgs = np.mean(time_stamps_slbfgs, axis=1)

# ssm
error_ssm = np.zeros((epoch_steffensen + 1, repeat))
time_stamps_ssm = np.zeros((epoch_steffensen + 1, repeat))
for i in range(repeat):
    w_path_ssm, time_stamp_ssm = alg.ssm(func_sto, initial_w, epoch_steffensen, m, b, n)
    for j in range(len(w_path_ssm)):
        error = func(w_path_ssm[j], 0) - values_opt
        error_ssm[j, i] = error
        time_stamps_ssm[j, i] = time_stamp_ssm[j]
error_ssm = np.mean(error_ssm, axis=1)
time_stamps_ssm = np.mean(time_stamps_ssm, axis=1)

# ssbb
error_ssbb = np.zeros((epoch_steffensen + 1, repeat))
time_stamps_ssbb = np.zeros((epoch_steffensen + 1, repeat))
for i in range(repeat):
    w_path_ssbb, time_stamp_ssbb = alg.ssbb(func_sto, initial_w, epoch_steffensen, m, b, n)
    for j in range(len(w_path_ssbb)):
        error = func(w_path_ssbb[j], 0) - values_opt
        error_ssbb[j, i] = error
        time_stamps_ssbb[j, i] = time_stamp_ssbb[j]
error_ssbb = np.mean(error_ssbb, axis=1)
time_stamps_ssbb = np.mean(time_stamps_ssbb, axis=1)

# normalized error vs passes
plt.semilogy(error_sgd, linewidth=2, label='SGD', marker='o', color='b')
plt.semilogy(error_sgd_bb, linewidth=2, label='SGD-BB', marker='^', color='g')
plt.semilogy(x_axis_svrg, error_svrg, linewidth=2, label='SVRG', marker='v', color='r')
plt.semilogy(x_axis_svrg, error_svrg_bb, linewidth=2, label='SVRG-BB', marker='s', color='c')
plt.semilogy(x_axis_slbfgs, error_slbfgs, linewidth=2, label='SLBFGS', marker='+', color='y')
plt.semilogy(x_axis_steffensen, error_ssm, linewidth=2, label='SSM', marker='x', color='m')
plt.semilogy(x_axis_steffensen, error_ssbb, linewidth=2, label='SSBB', marker='D', color='k')
plt.xlabel('passes through data')
plt.ylabel('log(suboptimality)')
plt.legend()
plt.xlim((0, 60))
plt.ylim((1e-16, 1))
plt.savefig('./plot/nonconvex/passes.png')
plt.show()

# normalized error vs time
plt.semilogy(time_stamps_sgd, error_sgd, linewidth=2, label='SGD', marker='o', color='b')
plt.semilogy(time_stamps_sgd_bb, error_sgd_bb, linewidth=2, label='SGD-BB', marker='^', color='g')
plt.semilogy(time_stamps_svrg, error_svrg, linewidth=2, label='SVRG', marker='v', color='r')
plt.semilogy(time_stamps_svrg_bb, error_svrg_bb, linewidth=2, label='SVRG-BB', marker='s', color='c')
plt.semilogy(time_stamps_slbfgs, error_slbfgs, linewidth=2, label='SLBFGS', marker='+', color='y')
plt.semilogy(time_stamps_ssm, error_ssm, linewidth=2, label='SSM', marker='x', color='m')
plt.semilogy(time_stamps_ssbb, error_ssbb, linewidth=2, label='SSBB', marker='D', color='k')
plt.xlabel('running time (s)')
plt.ylabel('log(suboptimality)')
plt.legend()
plt.xlim((0, 8))
plt.ylim((1e-16, 1))
plt.savefig('./plot/nonconvex/time.png')
plt.show()