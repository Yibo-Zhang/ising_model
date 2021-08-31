import time
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


def init_state(N):
    ''' generates a random spin configuration for initial condition'''
    # np.random.randint(2, size=(N,N)) 表示产生N*N随机数,返回为列表
    np.random.seed()
    state = 2*np.random.randint(2, size=(N, N))-1
    return state


def init_alpha(N):
    ''' generates a random connection parameter for initial condition'''
    # np.random.rand(2,N,N) 表示产生2*N*N随机0-1的数,返回为列表
    np.random.seed(2021)
    # -1 to 1 
    alpha = np.random.rand(2, N, N)
    return alpha


def flipping(grid,alpha):
    '''Monte Carlo move using Metropolis algorithm '''
    N = len(grid)
    for i in range(N):
        for j in range(N):
            # 产生0到N-1的随机数
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = grid[a][b]
            E = grid[(a+1) % N][b]*alpha[0][(a+1) % N][b] + \
                grid[a][(b+1) % N]*alpha[1][a][b] + \
                grid[(a-1) % N][b]*alpha[0][a][b] + \
                grid[a][(b-1) % N]*alpha[1][a][(b-1)]
            # -s-s*E == -2sE == cost
            cost = -2*s*E
            # 如果能量降低接受翻转
            if cost < 0:
                s *= -1
#             # 在0到1产生随机数，如果概率小于exp(-E/(kT))翻转
#             elif rand() < np.exp(-cost*beta):
#                 s *= -1
            grid[a][b] = s
    return grid


def calculate_energy(grid, alpha):
    '''Energy of a given configuration'''
    energy = 0
    N = len(grid)
    for i in range(N):
        for j in range(N):
            S = grid[i][j]
            E = grid[(i+1) % N][j]*alpha[0][(i+1) % N][j] + \
                grid[i][(j+1) % N]*alpha[1][i][j] + \
                grid[(i-1) % N][j]*alpha[0][i][j] + \
                grid[i][(j-1) % N]*alpha[1][i][(j-1) % N]
            # 负号来自哈密顿量
            energy += E*S
    # 最近邻4个格点
    return energy/4


def calculate_magnetic(grid):
    '''Magnetization of a given configuration'''
    mag = np.sum(grid)
    return mag

def simulation(grid_size = 10, step=3):
    N = 2**grid_size # 点阵尺寸, N x N
    step = 2**step
    Energy = []  # 内能
    Magnetization = []  # 磁矩
    # 开始模拟
    time_start = time.time()
    # 初始构型
    config = init_state(N)
    alpha = init_alpha(N)
    for i in range(step):
        config = flipping(config,alpha)
        e = calculate_energy(config,alpha)
        Energy.append(e)
        m = calculate_magnetic(config)
        Magnetization.append(m)
        if i % 300 == 0:
            print("已完成第%d步模拟" % i)
    time_end = time.time()
    print('totally cost', time_end-time_start)
    return Energy

if __name__ == '__main__':
    Energy_1 = simulation(grid_size=6, step=4)
    Energy_2 = simulation(grid_size=6, step=4)
    Energy_3 = simulation(grid_size=6, step=4)
    Energy_4 = simulation(grid_size=6, step=4)
    step = range(2**5)
    plt.plot(step, Energy_1, 'bo')
    plt.plot(step, Energy_2, 'r+')
    plt.plot(step, Energy_3, 'b*')
    plt.plot(step, Energy_4, 'g--')
    plt.xlabel('Step')
    plt.ylabel('Energy')
