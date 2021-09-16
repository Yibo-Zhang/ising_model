import time
import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl

### initialize state. 这里 设置了 theta 和 beta 两个角度，spin 在 x，y，z方向的投影分别为sin(theta)*cos(beta),
### sin(theta)*sin(beta) 和 cos(theta)
def init_theta(N):
    # (0-pi)
    return np.random.rand(N,N)*math.pi

def init_beta(N):
    # (0-2pi)
    return np.random.rand(N,N)*2*math.pi
def cal_cartesion(grid):
    theta,beta = grid[0,:,:], grid[1,:,:]
    x = np.sin(theta)*np.cos(beta)
    y = np.sin(theta)*np.sin(beta)
    z = np.cos(theta)  
    N = len(x)
    return np.concatenate([x,y,z],0).reshape([3,N,N]) 

def init_grid(N):
    #grid.shape = (2,N,N)
    theta = init_theta(N)
    theta = theta[np.newaxis, :]
    beta = init_beta(N)
    beta = beta[np.newaxis, :]
    return np.concatenate((theta,beta),axis=0)

def init_alpha(N):
    ''' generates a random connection parameter for initial condition'''
    # np.random.rand(2,N,N) 表示产生2*3*N*N随机0-1的数,返回为列表,# x,y,z 方向作用参数 一样
    np.random.seed(2021)
    # -1 to 1 
    alpha = 2*(np.random.rand(2,N, N)-0.5)
    return alpha


def calculate_single_energy(grid, a, b, alpha,lamda):
    # a,b 是 x 和 y的坐标
    N = alpha.shape[1]
    coordination = cal_cartesion(grid)
    s = coordination[:,a,b] # 选了一个点，包含x，y，z
    E = coordination[:,(a+1) % N,b]*alpha[0,(a+1) % N,b] + \
        coordination[:,a,(b+1) % N]*alpha[1,a,b] + \
        coordination[:,(a-1) % N,b]*alpha[0,a,b] + \
        coordination[:,a,(b-1) % N]*alpha[1,a,(b-1)%N]
    Energy = s*E
    single_point_energy = np.sum(Energy)/2+lamda*np.square(np.square(coordination[2,a,b])-1)
    return single_point_energy
def calculate_total_energy(grid,alpha,lamda):
    N = alpha.shape[1]
    total_energy = 0
    for a in range(N):
        for b in range(N):
            total_energy += calculate_single_energy(grid, a, b, alpha,lamda)
    return total_energy

#### fliping with perturbation

def calculate_perturbation_energy(grid,a,b,alpha,perturbation_theta,perturbation_beta,lamda):
    new_grid = np.copy(grid)
    new_grid[:,a,b] = new_grid[:,a,b]+[perturbation_theta,perturbation_beta]
    return calculate_single_energy(new_grid,a,b,alpha,lamda)
def flipping_perturbation(grid,alpha,lamda):
    '''Monte Carlo move using Metropolis algorithm '''
    N = grid.shape[1]
    for i in range(N):
        for j in range(N):
            # 产生0到N-1的随机数
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            perturbation_theta = (np.random.rand()-1)/2*math.pi # -0.25pi - 0.25pi
            perturbation_beta = (np.random.rand()-1)/2*math.pi # -0.25pi - 0.25pi
            origin_energy = calculate_single_energy(grid, a, b, alpha,lamda)
            later_energy = calculate_perturbation_energy(grid,a,b,alpha,perturbation_theta,perturbation_beta,lamda)
            cost = later_energy - origin_energy
            # 如果能量降低接受翻转
            if cost < 0:
                grid[:,a,b] = grid[:,a,b]+[perturbation_theta,perturbation_beta]
    return grid
###########
#### fliping with randomlized new direction

def calculate_new_direction_energy(grid,a,b,alpha,new_theta,new_beta,lamda):
    new_grid = np.copy(grid)
    new_grid[:,a,b] = [new_theta,new_beta]
    return calculate_single_energy(new_grid,a,b,alpha,lamda)

def flipping_random_new_direction(grid,alpha,lamda):
    '''Monte Carlo move using Metropolis algorithm '''
    N = grid.shape[1]
    for i in range(N):
        for j in range(N):
            # 产生0到N-1的随机数
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            new_theta = np.random.rand()*math.pi # -0.25pi - 0.25pi
            new_beta = np.random.rand()*2*math.pi # -0.25pi - 0.25pi
            origin_energy = calculate_single_energy(grid, a, b, alpha,lamda)
            later_energy = calculate_new_direction_energy(grid,a,b,alpha,new_theta,new_beta,lamda)
            cost = later_energy - origin_energy
            # 如果能量降低接受翻转
            if cost < 0:
                grid[:,a,b] = [new_theta,new_beta]
    return grid

######

#### simulation

def perturbation_simulation(grid_size = 10, step=3,lamda = 0):
    N = grid_size # 点阵尺寸, N x N
    Energy = []  # 内能
    # 开始模拟
    time_start = time.time()
    # 初始构型
    config = init_grid(N)
    alpha = init_alpha(N)
    for i in range(step):
        config = flipping_perturbation(config,alpha,lamda)
        e = calculate_total_energy(config,alpha,lamda=0)
        Energy.append(e)
        if i % 300 == 0:
            print("已完成第%d步模拟" % i)
    time_end = time.time()
    print('totally cost', time_end-time_start)
    return Energy,config,alpha
    
def new_direction_simulation(grid_size = 10, step=3,lamda = 0):
    N = grid_size # 点阵尺寸, N x N
    Energy = []  # 内能
    # 开始模拟
    time_start = time.time()
    # 初始构型
    config = init_grid(N)
    alpha = init_alpha(N)
    for i in range(step):
        config = flipping_random_new_direction(config,alpha,lamda = lamda)
        e = calculate_total_energy(config,alpha,lamda=0)
        Energy.append(e)
        if i % 300 == 0:
            print("已完成第%d步模拟" % i)
    time_end = time.time()
    print('totally cost', time_end-time_start)
    return Energy,config,alpha
##### plot spin
    
def plot_spin_z(config):
    N = config.shape[1]
    cartesion = cal_cartesion(config)
    z_direction_config = cartesion[2,:,:]
    x_position = np.linspace(0,1,N)
    y_position = np.linspace(0,1,N)
    m_x = np.zeros([N,N])
    m_y = z_direction_config

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

# if __name__ == '__main__':
#     step = 3
#     grid_size = 2
#     Energy_1,config1,alpha1 = simulation(grid_size=grid_size, step=step)
#     Energy_2,config2,alpha2 = simulation(grid_size=grid_size, step=step)
#     Energy_3,config3,alpha3 = simulation(grid_size=grid_size, step=step)
#     Energy_4,config4,alpha4 = simulation(grid_size=grid_size, step=step)

#     step = range(2**step)
#     site_num = (grid_size)**2
#     energy1_average = [i/site_num for i in Energy_1]
#     energy2_average = [i/site_num for i in Energy_2]
#     energy3_average = [i/site_num for i in Energy_3]
#     energy4_average = [i/site_num for i in Energy_4]

#     ### plot config
#     plot_spin(config1)
#     plot_spin(config2)
#     plot_spin(config3)
#     plot_spin(config4)

#     ### plot energy vs step
#     plt.plot(step, Energy_1, 'bo')
#     plt.plot(step, Energy_2, 'r+')
#     plt.plot(step, Energy_3, 'b*')
#     plt.plot(step, Energy_4, 'g--')
#     plt.xlabel('Step')
#     plt.ylabel('Energy')

#     ### plot average energy vs step
#     plt.plot(step, energy1_average, 'bo')
#     plt.plot(step, energy2_average, 'r+')
#     plt.plot(step, energy3_average, 'b*')
#     plt.plot(step, energy4_average, 'g--')
#     plt.xlabel('Step')
#     plt.ylabel('Energy')
