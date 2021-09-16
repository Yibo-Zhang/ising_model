import time
import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl

class Perturbation_model:
    def __init__(self,N,step,label='perturbation model'):
        self.N = N
        self.step=step
        self.config = self.init_grid(N)
        self.alpha = self.init_alpha(N)
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

    def init_alpha(self,N):
        ''' generates a random connection parameter for initial condition'''
        # np.random.rand(2,N,N) 表示产生2*3*N*N随机0-1的数,返回为列表,# x,y,z 方向作用参数 一样
        np.random.seed(2021)
        # -1 to 1 
        alpha = 2*(np.random.rand(2,N, N)-0.5)
        np.random.seed()

        return alpha
    def calculate_single_energy(self,grid, a, b, alpha,lamda):
        # a,b 是 x 和 y的坐标
        N = self.N
        coordination = self.cal_cartesion(grid)
        s = coordination[:,a,b] # 选了一个点，包含x，y，z
        E = coordination[:,(a+1) % N,b]*alpha[0,(a+1) % N,b] + \
            coordination[:,a,(b+1) % N]*alpha[1,a,b] + \
            coordination[:,(a-1) % N,b]*alpha[0,a,b] + \
            coordination[:,a,(b-1) % N]*alpha[1,a,(b-1)%N]
        Energy = s*E
        single_point_energy = np.sum(Energy)/2+lamda*np.square(np.square(coordination[2,a,b])-1)
        return single_point_energy
    def calculate_total_energy(self,grid,alpha,lamda):
        N = self.N
        total_energy = 0
        for a in range(N):
            for b in range(N):
                total_energy += self.calculate_single_energy(grid, a, b, alpha,lamda)
        return total_energy
    def calculate_perturbation_energy(self,grid,a,b,alpha,perturbation_theta,perturbation_beta,lamda):
        new_grid = np.copy(grid)
        new_grid[:,a,b] = new_grid[:,a,b]+[perturbation_theta,perturbation_beta]
        return self.calculate_single_energy(new_grid,a,b,alpha,lamda)
    def flipping_perturbation(self,lamda):
        '''Monte Carlo move using Metropolis algorithm '''
        for i in range(self.N):
            for j in range(self.N):
                # 产生0到N-1的随机数
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                perturbation_theta = (np.random.rand()-1)/2*math.pi # -0.25pi - 0.25pi
                perturbation_beta = (np.random.rand()-1)/2*math.pi # -0.25pi - 0.25pi
                origin_energy = self.calculate_single_energy(self.config, a, b, self.alpha,lamda)
                later_energy = self.calculate_perturbation_energy(self.config,a,b,self.alpha,perturbation_theta,perturbation_beta,lamda)
                cost = later_energy - origin_energy
                # 如果能量降低接受翻转
                if cost < 0:
                    self.config[:,a,b] = self.config[:,a,b]+[perturbation_theta,perturbation_beta]
    def simulation(self,lamda = 0):
        Energy = []  # 内能
        # 开始模拟
        time_start = time.time()
        # 初始构型
        for i in range(self.step):
            self.flipping_perturbation(lamda)
            e = self.calculate_total_energy(self.config,self.alpha,lamda=0)
            Energy.append(e)
            if i % 300 == 0:
                print("已完成第%d步模拟" % i)
        time_end = time.time()
        print('totally cost', time_end-time_start)
        average_energy = [i/np.square(self.N) for i in Energy]
        plt.plot(average_energy)
        plt.plot(average_energy,label=self.label)
        plt.legend()


class New_direction_model:
    def __init__(self,N,step,label='new direction model'):
        self.N = N
        self.step=step
        self.config = self.init_grid(N)
        self.alpha = self.init_alpha(N)
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

    def init_alpha(self,N):
        ''' generates a random connection parameter for initial condition'''
        # np.random.rand(2,N,N) 表示产生2*3*N*N随机0-1的数,返回为列表,# x,y,z 方向作用参数 一样
        np.random.seed(2021)
        # -1 to 1 
        alpha = 2*(np.random.rand(2,N, N)-0.5)
        np.random.seed()
        return alpha
    def calculate_single_energy(self,grid, a, b, alpha,lamda):
        # a,b 是 x 和 y的坐标
        N = self.N
        coordination = self.cal_cartesion(grid)
        s = coordination[:,a,b] # 选了一个点，包含x，y，z
        E = coordination[:,(a+1) % N,b]*alpha[0,(a+1) % N,b] + \
            coordination[:,a,(b+1) % N]*alpha[1,a,b] + \
            coordination[:,(a-1) % N,b]*alpha[0,a,b] + \
            coordination[:,a,(b-1) % N]*alpha[1,a,(b-1)%N]
        Energy = s*E
        single_point_energy = np.sum(Energy)/2+lamda*np.square(np.square(coordination[2,a,b])-1)
        return single_point_energy
    def calculate_total_energy(self,config,alpha,lamda):
        N = self.N
        total_energy = 0
        for a in range(N):
            for b in range(N):
                total_energy += self.calculate_single_energy(config, a, b, alpha,lamda)
        return total_energy
    def calculate_new_direction_energy(self,grid,a,b,alpha,new_theta,new_beta,lamda):
        new_grid = np.copy(grid)
        new_grid[:,a,b] = [new_theta,new_beta]
        return self.calculate_single_energy(new_grid,a,b,alpha,lamda)

    def flipping_random_new_direction(self,lamda):
        '''Monte Carlo move using Metropolis algorithm '''
        for i in range(self.N):
            for j in range(self.N):
                # 产生0到N-1的随机数
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                new_theta = np.random.rand()*math.pi # -0.25pi - 0.25pi
                new_beta = np.random.rand()*2*math.pi # -0.25pi - 0.25pi
                origin_energy = self.calculate_single_energy(self.config, a, b, self.alpha,lamda)
                later_energy = self.calculate_new_direction_energy(self.config,a,b,self.alpha,new_theta,new_beta,lamda)
                cost = later_energy - origin_energy
                # 如果能量降低接受翻转
                if cost < 0:
                    self.config[:,a,b] = [new_theta,new_beta]
    def simulation(self,lamda = 0):
        Energy = []  # 内能
        # 开始模拟
        time_start = time.time()
        for i in range(self.step):
            self.flipping_random_new_direction(lamda = lamda)
            e = self.calculate_total_energy(self.config,self.alpha,lamda=0)
            Energy.append(e)
            if i % 300 == 0:
                print("已完成第%d步模拟" % i)
        time_end = time.time()
        print('totally cost', time_end-time_start)
        average_energy = [i/np.square(self.N) for i in Energy]
        self.energy = Energy
        plt.plot(average_energy,label=self.label)
        plt.legend()
    def plot_spin_z(self):
        N = self.N
        cartesion = self.cal_cartesion(self.config)
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


    
class Up_down_flip_model(New_direction_model):
    def __init__(self,N,step,label='simple flip model'):
        super().__init__(N,step,label)
    def init_grid(self,N):
        theta = np.random.randint(2,size=(N,N))*math.pi
        theta = theta[np.newaxis, :]
        beta = np.zeros((N,N))
        beta = beta[np.newaxis, :]
        return np.concatenate((theta,beta),axis=0)

    def flipping_random_new_direction(self,lamda):
        '''Monte Carlo move using Metropolis algorithm '''
        for i in range(self.N):
            for j in range(self.N):
                # 产生0到N-1的随机数
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                new_theta = np.random.randint(2)*math.pi # 0 to pi
                new_beta = 0 # 0
                origin_energy = super().calculate_single_energy(self.config, a, b, self.alpha,lamda)
                later_energy = super().calculate_new_direction_energy(self.config,a,b,self.alpha,new_theta,new_beta,lamda)
                cost = later_energy - origin_energy
                # 如果能量降低接受翻转
                if cost < 0:
                    self.config[:,a,b] = [new_theta,new_beta]
    

