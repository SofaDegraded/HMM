from PIL import Image
import numpy as np
import scipy as sp
from functools import reduce
import matplotlib.pyplot as plt
import os
def get_obs(path, K):
    pix_bit = []
    step = 5
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    im = [Image.open(imp).convert('1') for imp in image_paths]
    for k in range(K):
        W = im[k].width
        H = im[k].height
        pix = []
        for i in range(step, H, step - 1):
            for j in range(0, W, step - 1):
                if j + step < W:
                    GP = im[k].crop((j, i - step, j + step,i))
                    x1 = GP._ImageCrop__crop[0]
                    x3 = GP._ImageCrop__crop[2]
                    y1 = GP._ImageCrop__crop[1]
                    y3 = GP._ImageCrop__crop[3]
                    x = (x1 + x3) / 2
                    y = (y1 + y3) / 2
                    pix.append(im[k].getpixel((x,y)))
        pix_bit.append(list(map(lambda x: 1 if x == 255 else x, pix)))
    return np.array(pix_bit)
def get_data(fname, type):
    O = np.array([[i for i in line.split()] for line in open(fname, encoding="utf-8")], dtype=type)
    return O

def get_data1(fname, type):
    O = np.array([i for i in open(fname, encoding="utf-8").readline().split()], dtype=type)
    return O

def WritingInFile(names, sequences, fileName):
    with open(fileName, "w") as file:
        for line in sequences:
            print(line, file=file)

def forward_path(O, pi, A, B, T, N, K):
    alpha_k = []
    for k in range(K):
        alpha = np.zeros((T, N))
        alpha[0, :] = pi * B[:, O[k, 0]]
        for t in range(1, T):
            for j in range(N):
                tmp = np.zeros(N)
                for i in range(N):
                    tmp[i] = alpha[t - 1, i] * A[i, j]
                alpha[t, j] = tmp.sum() * B[j, O[k, t]]
        alpha_k.append(alpha)
    return np.array(alpha_k)

def backward_path(O, pi, A, B, T, N, K):
    beta_k = []
    for k in range(K):
        beta = np.zeros((T, N))
        beta[T - 1, :] = 1
        for t in range(T - 2, -1, -1):
            for i in range(N):
                tmp = np.zeros(N)
                for j in range(N):
                    tmp[j] = beta[t + 1, j] * A[i, j] * B[j, O[k, t + 1]]
                beta[t, i] = tmp.sum()
        beta_k.append(beta)
    return np.array(beta_k)

def calculate_gamma(alpha, beta, T, N, K):
    gamma_k = []
    for k in range(K):
        gamma = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                gamma[t, i] = alpha[k, t, i] * beta[k, t, i]
            sum_all = gamma[t, :].sum()
            gamma[t, :] = gamma[t, :] / sum_all
        gamma_k.append(gamma)
    return np.array(gamma_k)

def calculate_ksi(O, alpha, beta, A, B, T, N, K):
    ksi_k = []
    for k in range(K):
        ksi = np.zeros((T, N, N))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    ksi[t, i, j] = alpha[k, t, i] * A[i, j] * beta[k, t + 1, j] * B[j, O[k, t + 1]]
            sum_all = ksi[t, :, :].sum()
            ksi[t, :, :] = ksi[t, :, :] / sum_all
        ksi_k.append(ksi)
    return np.array(ksi_k)

def estimate_parameter(O, pi_0, A_0, B_0, T, N, M, K):
    alp = forward_path(O, pi_0, A_0, B_0, T, N, K)
    bet = backward_path(O, pi_0, A_0, B_0, T, N, K)
    gam = calculate_gamma(alp, bet, T, N, K)
    ksi = calculate_ksi(O, alp, bet, A_0, B_0, T, N, K)

    est_pi = np.sum(gam[:, 0, :], axis=0) / K

    est_A_k = np.zeros((K, N, N))
    for k in range(K):
        for i in range(N):
            denom = gam[k, :-1, i].sum()
            for j in range(N):
                est_A_k[k, i, j] = ksi[k, :-1, i, j].sum() / denom

    est_A = np.sum(est_A_k, axis=0) / K
    est_B_k = np.zeros((K, N, M))
    for k in range(K):
        for i in range(N):
            denom = gam[k, :, i].sum()
            for j in range(M):
                numer = gam[k, :, i][O[k] == j].sum()
                est_B_k[k, i, j] = numer / denom
    est_B = np.sum(est_B_k, axis=0) / K
    return est_pi, est_A, est_B

def log_likelihood(O, pi, A, B, T, N, K):
    alp = forward_path(O, pi, A, B, T, N, K)
    L = []
    for k in range(K):
        l = np.zeros((N))
        for i in range(N):
            l[i] = alp[k, T - 1, i]
        sum_all = l[:].sum()
        L.append(sum_all)
    lnL = np.sum(np.log(L))
    return lnL

def forward_path1(O, pi, A, B, T, N, K):
    alpha = np.zeros((T, N))
    alpha[0, :] = pi * B[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            tmp = np.zeros(N)
            for i in range(N):
                tmp[i] = alpha[t - 1, i] * A[i, j]
            alpha[t, j] = tmp.sum() * B[j, O[t]]
    return np.array(alpha)
def log_likelihood_for_learn_or_test(O, pi, A, B, T, N):
    alp = forward_path1(O, pi, A, B, T, N, 1)
    l = np.zeros((N))
    for i in range(N):
        l[i] = alp[T - 1, i]
    L = l[:].sum()
    lnL = np.sum(np.log(L))
    return lnL

def iter_exit(O, pi_old, A_old, B_old, pi_new, A_new, B_new, T, N, K):
    old = log_likelihood(O, pi_old, A_old, B_old, T, N, K)
    new = log_likelihood(O, pi_new, A_new, B_new, T, N, K)
    exit = abs(old - new)
    if exit > 1e-3:
        return False, exit
    else:
        return True, exit

def baum_welch(O, pi, A, B, T, N, M, K):
    iter = 0
    exit = False
    max_iter = 100
    ex = []
    temp = []
    temp.append(log_likelihood(O, pi, A, B, T, N, K))
    while exit == False:
        iter += 1
        new_pi, new_A, new_B = estimate_parameter(O, pi, A, B, T, N, M, K)
        exit, tmp = iter_exit(O, pi, A, B, new_pi, new_A, new_B, T, N, K)
        temp.append(log_likelihood(O, new_pi, new_A, new_B, T, N, K))
        if iter > max_iter:
            exit = True
        ex.append(tmp)
        pi, A, B = new_pi, new_A, new_B
    return pi, A, B, ex

def test_or_learn():
    K = 10      #количество картинок в каждом классе для обучения
    N = 2       #число скрытых состояний
    M = 2       #алфавит
    CL = 5      #число классов
    path = []
    #path_test = []
    path.append(os.path.abspath('./1/'))
    path.append(os.path.abspath('./2/'))
    path.append(os.path.abspath('./3/'))
    path.append(os.path.abspath('./4/'))
    path.append(os.path.abspath('./5/'))
    path_test.append(os.path.abspath('./6/'))
    path_test.append(os.path.abspath('./7/'))
    path_test.append(os.path.abspath('./8/'))
    path_test.append(os.path.abspath('./9/'))
    path_test.append(os.path.abspath('./10/'))
    A_path = os.path.abspath('./A/')
    B_path = os.path.abspath('./B/')
    pi_path = os.path.abspath('./PI/')
    A_arr = [os.path.join(A_path, f) for f in os.listdir(A_path) if f.endswith('.txt')]
    B_arr = [os.path.join(B_path, f) for f in os.listdir(B_path) if f.endswith('.txt')]
    pi_arr = [os.path.join(pi_path, f) for f in os.listdir(pi_path) if f.endswith('.txt')]
    A_0 = [get_data(a, np.double) for a in A_arr]
    B_0 = [get_data(b, np.double) for b in B_arr]
    pi_0 = [get_data(pi, np.double) for pi in pi_arr]
    O = [get_obs(el, K) for el in path]
    O_test = [get_obs(el, K) for el in path_test]
    #оценка параметров
    est_p = []
    for k in range(CL):#количество классов
        est = []
        lnL = []
        for i in range(5):#количество начальных приближений
            est_pi, est_A, est_B, ex = baum_welch(O[k], np.array(pi_0[i]), np.array(A_0[i]), np.array(B_0[i]), O[k].shape[1], N, M, K)
            est.append([est_pi, est_A, est_B])
            lnL.append(log_likelihood(O[k], est_pi, est_A, est_B, O[k].shape[1], N, K))
        maxlnL = lnL.index(np.max(np.array(lnL)))
        est_p.append(est[maxlnL])
    WritingInFile(['est'], est_p, 'est.txt')
    
    test_t = []
    for c in range(5):#class
        test_t.append([])
        for i in range(K):
            test_t[c].append([])
            for j in range(5):#class
                test_t[c][i].append(log_likelihood_for_learn_or_test(O_test[c][i], est_p[j][0], est_p[j][1], est_p[j][2], O_test[c][i].shape[0], N))
    res_class_t = np.argmax(test_t, axis = 2)
    WritingInFile(['rest'], res_class_t, 'rest.txt')
    return est_p

test_or_learn()
