import time
import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl


class New_direction_model:
    def __init__(self,N,step,label='new direction model'):
        self.N = N
        self.step=step
        self.config = self.init_grid(N)
        self.J = self.init_J(N)
        self.label = label
    ### initialize state. 这里 设置了 theta 和 beta 两个角度，spin 在 x，y，z方向的投影分别为sin(theta)*cos(beta),
    ### sin(theta)*sin(beta) 和 cos(theta)
    def init_theta(self,N):
        # (0-pi)
        return np.random.rand(N,N)*math.pi
    def init_beta(self,N):
        # (0-2pi)
        return np.random.rand(N,N)*2*math.pi
    def cal_cartesion(self,grid):
        theta,beta = grid[0,:,:], grid[1,:,:]
        x = np.sin(theta)*np.cos(beta)
        y = np.sin(theta)*np.sin(beta)
        z = np.cos(theta)  
        N = self.N
        return np.concatenate([x,y,z],0).reshape([3,N,N]) 

    def init_grid(self,N):
        #grid.shape = (2,N,N)
        theta = self.init_theta(N)
        theta = theta[np.newaxis, :]
        beta = self.init_beta(N)
        beta = beta[np.newaxis, :]
        return np.concatenate((theta,beta),axis=0)

    def init_J(self,N):
        ''' generates a random connection parameter for initial condition'''
        # np.random.rand(2,N,N) 表示产生2*3*N*N随机0-1的数,返回为列表,# x,y,z 方向作用参数 一样
        np.random.seed(2021)
        # -1 to 1 
        J = 2*(np.random.rand(2,N, N)-0.5)
        np.random.seed()
        return J
    def calculate_single_energy(self,grid, a, b, J,lamda):
        # a,b 是 x 和 y的坐标
        N = self.N
        coordination = self.cal_cartesion(grid)
        s = coordination[:,a,b] # 选了一个点，包含x，y，z
        E = coordination[:,(a+1) % N,b]*J[0,(a+1) % N,b] + \
            coordination[:,a,(b+1) % N]*J[1,a,b] + \
            coordination[:,(a-1) % N,b]*J[0,a,b] + \
            coordination[:,a,(b-1) % N]*J[1,a,(b-1)%N]
        Energy = s*E
        single_point_energy = np.sum(Energy)/2+lamda*np.square(np.square(coordination[2,a,b])-1)
        return single_point_energy
    def calculate_total_energy(self,config,lamda):
        N = self.N
        total_energy = 0
        for a in range(N):
            for b in range(N):
                total_energy += self.calculate_single_energy(config, a, b, self.J,lamda)
        return total_energy
    def calculate_new_direction_energy(self,grid,a,b,J,new_theta,new_beta,lamda):
        new_grid = np.copy(grid)
        new_grid[:,a,b] = [new_theta,new_beta]
        return self.calculate_single_energy(new_grid,a,b,J,lamda)

    def flipping_random_new_direction(self,lamda):
        '''Monte Carlo move using Metropolis algorithm '''
        for i in range(self.N):
            for j in range(self.N):
                # 产生0到N-1的随机数
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                new_theta = np.random.rand()*math.pi # 0 - pi
                new_beta = np.random.rand()*2*math.pi # 0 - 2pi
                origin_energy = self.calculate_single_energy(self.config, a, b, self.J,lamda)
                later_energy = self.calculate_new_direction_energy(self.config,a,b,self.J,new_theta,new_beta,lamda)
                cost = later_energy - origin_energy
                # 如果能量降低接受翻转
                if cost < 0:
                    self.config[:,a,b] = [new_theta,new_beta]
    def simulation(self,lamda = 0):
        Energy = []  # 内能
        # 开始模拟
        self.lowest_energy_with_J = np.copy(self.calculate_total_energy(self.config,lamda=lamda))
        self.lowest_energy = np.copy(self.calculate_total_energy(self.config,lamda=0))
        self.lowest_config = np.copy(self.config)
        time_start = time.time()
        for i in range(self.step):
            if i==np.rint(self.step/5):
                lamda = 0.5
            if i == np.rint(2*self.step/5):
                lamda = 1
            if i==np.rint(3*self.step/5):
                lamda = 1.5
            if i == np.rint(4*self.step/5):
                lamda = 2
                
            self.flipping_random_new_direction(lamda = lamda)
            e = self.calculate_total_energy(self.config,lamda=lamda)
            if e < self.lowest_energy_with_J:
                self.lowest_energy_with_J = np.copy(self.calculate_total_energy(self.config,lamda=lamda))
                self.lowest_energy = np.copy(np.copy(self.calculate_total_energy(self.config,lamda=0)))
                self.lowest_config = np.copy(self.config)
            Energy.append(e)
            if i % 300 == 0:
                print("已完成第%d步模拟" % i)
        # 把三维的spin 转换成 up down
        self.convert_config_to_up_down()
        # print total spin energy
        self.lowest_energy = self.calculate_total_energy(self.lowest_config,lamda=0)
        time_end = time.time()
        print('totally cost', time_end-time_start)
        average_energy = [i/np.square(self.N) for i in Energy]
        self.energy = Energy
        self.average_energy = average_energy
        plt.plot(average_energy,label=self.label)
        print()
        plt.legend()
    def convert_config_to_up_down(self):
        self.lowest_config[0,:,:][self.lowest_config[0,:,:]<(math.pi/2)] = 0
        self.lowest_config[0,:,:][self.lowest_config[0,:,:]>(math.pi/2)] = math.pi
    def plot_spin_z(self):
        N = self.N
        cartesion = self.cal_cartesion(self.lowest_config)
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