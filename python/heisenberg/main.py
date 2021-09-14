import time
import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl

### initialize state. 这里 设置了 theta 和 beta 两个角度，spin 在 x，y，z方向的投影分别为cos(theta)*cos(beta),
### cos(theta)*sin(beta) 和 sin(theta)
def init_theta(N):
    # (0-2pi)
    return np.random.rand(N,N)*2*math.pi

def init_beta(N):
    # (0-2pi)
    return np.random.rand(N,N)*2*math.pi
def cal_projection(grid):
    theta,beta = grid[0,:,:], grid[1,:,:]
    x = np.cos(theta)*np.cos(beta)
    y = np.cos(theta)*np.sin(beta)
    z = np.sin(theta)  
    return x,y,z 

def init_grid(N):
    (2,N,N)
    theta = init_theta(N)
    theta = theta[np.newaxis, :]
    beta = init_beta(N)
    beta = beta[np.newaxis, :]
    return np.concatenate((theta,beta),axis=0)

def init_alpha(N):
    ''' generates a random connection parameter for initial condition'''
    # np.random.rand(2,3,N,N) 表示产生2*3*N*N随机0-1的数,返回为列表,# 3 表示 x,y,z 方向的相互作用
    np.random.seed(2021)
    # -1 to 1 
    alpha = 2*(np.random.rand(2, 3,N, N)-0.5)
    return alpha


def calculate_single_energy(grid, x_point, y_point, alpha):
    x, y, z = cal_projection(grid)
    x_alpha = alpha[:,0,:,:]
    x_s = x[x_point][y_point]
    x_E = x[(a+1) % N][b]*alpha[0][(a+1) % N][b] + \
        x[a][(b+1) % N]*alpha[1][a][b] + \
        x[(a-1) % N][b]*alpha[0][a][b] + \
        x[a][(b-1) % N]*alpha[1][a][(b-1)]
    x_cost = -2*x_s*x_E



    s = grid[a][b]
    E = grid[(a+1) % N][b]*alpha[0][(a+1) % N][b] + \
        grid[a][(b+1) % N]*alpha[1][a][b] + \
        grid[(a-1) % N][b]*alpha[0][a][b] + \
        grid[a][(b-1) % N]*alpha[1][a][(b-1)]
    cost = -2*s*E





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
            # 在0到1产生随机数，如果概率小于exp(-E/(kT))翻转
            # elif np.random.rand() < np.exp(-cost/temperature):
            #     s *= -1
            grid[a][b] = s
    return grid



def calculate_energy(grid, alpha):
    '''Energy of a given configuration'''
    energy = 0
    N = len(grid[0])
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




def simulation(grid_size = 10, step=3):
    N = grid_size # 点阵尺寸, N x N
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
    return Energy,config,alpha
def cluster_flip_simulation(grid_size=10, step=3):
    N = grid_size # 点阵尺寸, N x N
    step = 2**step
    Energy = []  # 内能
    Magnetization = []  # 磁矩
    # 开始模拟
    time_start = time.time()
    # 初始构型
    config = init_state(N)
    alpha = init_alpha(N)
    for i in range(step):
        config = clusterFlip(config,alpha)
        e = calculate_energy(config,alpha)
        Energy.append(e)
        m = calculate_magnetic(config)
        Magnetization.append(m)
        if i % 300 == 0:
            print("已完成第%d步模拟" % i)
    time_end = time.time()
    print('totally cost', time_end-time_start)
    return Energy,config,alpha

def plot_spin(config):
    N = len(config)
    x_position = np.linspace(0,1,N)
    y_position = np.linspace(0,1,N)
    m_x = np.zeros([N,N])
    m_y = config
    #     m_z = m_z.numpy()

    x_position,y_position = np.meshgrid(x_position, y_position)
    plt.figure(figsize=(N,N))
    ax = plt.subplot(1, 1, 1)

    color = m_y
    map_range=[-1, 1]
    norm = mpl.colors.Normalize(vmin=map_range[0], vmax=map_range[1])
    colormap = mpl.cm.bwr
    color_map = colormap(norm(color))
    color_map = color_map.reshape([-1, 4])
    
    quiver = ax.quiver(x_position, y_position, m_x, m_y,color=color_map,
                       angles='xy', pivot='mid', scale=10)

if __name__ == '__main__':
    step = 3
    grid_size = 2
    Energy_1,config1,alpha1 = simulation(grid_size=grid_size, step=step)
    Energy_2,config2,alpha2 = simulation(grid_size=grid_size, step=step)
    Energy_3,config3,alpha3 = simulation(grid_size=grid_size, step=step)
    Energy_4,config4,alpha4 = simulation(grid_size=grid_size, step=step)

    step = range(2**step)
    site_num = (grid_size)**2
    energy1_average = [i/site_num for i in Energy_1]
    energy2_average = [i/site_num for i in Energy_2]
    energy3_average = [i/site_num for i in Energy_3]
    energy4_average = [i/site_num for i in Energy_4]

    ### plot config
    plot_spin(config1)
    plot_spin(config2)
    plot_spin(config3)
    plot_spin(config4)

    ### plot energy vs step
    plt.plot(step, Energy_1, 'bo')
    plt.plot(step, Energy_2, 'r+')
    plt.plot(step, Energy_3, 'b*')
    plt.plot(step, Energy_4, 'g--')
    plt.xlabel('Step')
    plt.ylabel('Energy')

    ### plot average energy vs step
    plt.plot(step, energy1_average, 'bo')
    plt.plot(step, energy2_average, 'r+')
    plt.plot(step, energy3_average, 'b*')
    plt.plot(step, energy4_average, 'g--')
    plt.xlabel('Step')
    plt.ylabel('Energy')
