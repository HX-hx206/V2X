# 修改奖励函数
from __future__ import division
import numpy as np
import time
import random
import math
# from scipy.special import jv  # 贝塞尔函数
from scipy.special import j0
import scipy
import os

class HumanDrivenVehicle:
    def __init__(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
        self.agent_id = agent_id                                # ID
        self.max_power = max_power                              # 最大发射功率
        self.current_power = np.random.rand() * max_power       # 当前发射功率
        self.position = start_position                          # 当前位置
        self.direction = start_direction                        # 当前方向
        self.velocity = velocity                                # 速度
        self.neighbors = []                                     # 邻居
        self.destinations = []                                  # 通信目的地
        self.type = type_id                                     # 有人驾驶或者无人驾驶

    def take_action(self):
        # 随机选择一个发射功率,后根据策略更新
        self.current_power = np.random.rand() * self.max_power
        return self.current_power

class AutonomousDrivenVehicle:
    def __init__(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
        self.agent_id = agent_id                                # ID
        self.max_power = max_power                              # 最大发射功率
        self.current_power = np.random.rand() * max_power       # 当前发射功率
        self.position = start_position                          # 当前位置
        self.direction = start_direction                        # 当前方向
        self.velocity = velocity                                # 速度
        self.neighbors = []                                     # 邻居
        self.destinations = []                                  # 通信目的地
        self.type = type_id                                     # 是否为自动控制

    def take_action(self):
        # 随机选择一个发射功率，后根据策略更新
        self.current_power = np.random.rand() * self.max_power
        return self.current_power

class V2Vchannels:
    def __init__(self):
        self.t = 0                              # 时间
        self.h_bs = 1.5                         # 基站高度
        self.h_ms = 1.5                         # 移动设备高度
        self.fc = 2                             # 频率（GHz）
        self.decorrelation_distance = 10        # 去相关距离（m）
        self.shadow_std = 3                     # 阴影标准差
        self.G = 70                             # 路径损失常数
        self.alpha = 3                         # 路径损耗的指数
        self.c = 3e8                            # 光速（m/s）
        self.update_time_interval = 0.1         # 时间步长（s）
        # self.f_D = self.get_max_doppler_shift(velocity1,velocity2) # 最大多普勒频移
        # self.epsilon = self.get_epsilon()       # 小尺度衰落系数 ε

    def get_distance(self, position_A, position_B):


    '''计算大尺度衰落'''
    def get_shadowing(self, delta_distance, shadowing):                                 # delta_distance车辆间相对移动距离

    def get_large_scale_fading(self, position_A, position_B, shadowing):


class V2Ichannels:
    def __init__(self):
        self.t = 0                                  # 时间
        self.h_bs = 25                              # 基站高度
        self.h_ms = 1.5                             # 移动设备高度
        self.fc = 2                                 # 频率（GHz）
        self.decorrelation_distance = 50            # 去相关距离
        self.BS_position = [750 / 2, 1299 / 2]      # 基站位置
        self.shadow_std = 8                         # 阴影标准差
        self.G = 70                             # 路径损失常数
        self.alpha = 3                             # 路径损耗的指数
        self.c = 3e8                                # 光速（m/s）
        self.v = 50                                 # 车速（m/s）
        self.update_time_interval = 0.1             # 时间步长（s）
        # self.f_D = self.get_max_doppler_shift()     # 最大多普勒频移
        # self.epsilon = self.get_epsilon()           # 小尺度衰落系数 ε


    def get_distance(self, position_A):
        """计算两点间的距离"""
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        return math.hypot(d1, d2) + 0.001  # 避免距离为 0

    '''计算大尺度衰落'''
    def get_shadowing(self, delta_distance, shadowing):
        """计算对数正态分布的阴影衰落增益"""
        # shadowings = np.array(shadowing)
        shadowing_dB = (
                np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing +
                math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) *
                np.random.normal(0, self.shadow_std)
        )
        shadowing_linear = 10 ** (shadowing_dB / 10)  # 转换为线性增益
        return shadowing_linear

    def get_large_scale_fading(self, position_A, shadowing):
        """计算大尺度衰落 L_{x,y}(t)"""
        d = self.get_distance(position_A)  # 计算距离
        shadowing_linear = self.get_shadowing(d, shadowing)  # 计算阴影衰落
        large_scale_fading = self.G * shadowing_linear / (d ** self.alpha)  # 大尺度衰落公式
        return large_scale_fading

class Environ:
        # 初始化需要传入4个list（为上下左右路口的位置数据）：down_lane, up_lane, left_lane, right_lane；地图的宽和高；
        # 车辆数和邻居数。除以上所提外，内部含有好多参数，如下：
        def __init__(self, down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor):

            self.down_lanes = down_lanes            #实例化
            self.up_lanes = up_lanes                #实例化
            self.left_lanes = left_lanes            #实例化
            self.right_lanes = right_lanes          #实例化
            self.width = width                      #实例化
            self.height = height                    #实例化

            '''信道和车辆定义'''
            self.V2Vchannels = V2Vchannels()        #实例化
            self.V2Ichannels = V2Ichannels()        #实例化
            self.HumanDrivenVehicle_list = []            #初始化列表
            self.AutonomousDrivenVehicle_list = []       #初始化列表
            self.vehicles = []                      #初始化车辆列表

            '''状态和信道参数'''
            self.demand = []                        #需求（数据）
            self.V2V_Shadowing = []                 #V2V通信的阴影衰落
            self.V2I_Shadowing = []                 #V2I通信的阴影衰落
            self.delta_distance = []                #车辆之间的距离
            self.V2V_channels_abs = []              #V2V信道绝对值
            self.V2I_channels_abs = []              #V2I信道绝对值

            '''通信功率和噪声'''
            self.update_time_interval = 0.1         #时间步长（s）
            self.V2I_power_dB = 46                  #V2I通信功率
            self.V2V_power_dB_List = [23, 20, 8, 5,-100]
            self.sig2_dB = -114                     #噪声功率
            self.bsAntGain = 10                     #基站天线增益
            self.bsNoiseFigure = 5                  #基站噪声系数
            self.vehAntGain = 3                     #车辆天线增益
            self.vehNoiseFigure = 9                 #车辆噪声系数
            self.sig2 = 10 ** (self.sig2_dB / 10)   #dB转化成mW单位
            self.f_c = 2e9                          #载波频率 (Hz)
            self.c = 3e8                            #光速 (m/s)


            '''资源块和车辆设置'''
            self.n_RB = n_veh * 2                       #资源块数设置为车辆数
            # self.n_RB = 1                           # 资源块数设置为车辆数
            self.n_Veh = n_veh * 2                      #模拟中车辆数
            self.n_neighbor = n_neighbor            #每辆车可通信的临近车辆数

            '''时间和带宽设置'''
            self.time_fast = 0.001                  #快速衰落持续时间1ms
            self.time_slow = 0.1                    #慢衰落持续时间100ms
            self.bandwidth = int(1e6)               #通信带宽设置为1MHz

            '''需求和干扰设置'''
            self.demand_size = int((4 * 190 + 300) * 8 * 1)  #设置通信需求的大小（以bits为单位）。根据车辆数量和一些特定需求大小来计算的。
            self.V2V_Interference_all = np.zeros((self.n_Veh * 2, self.n_neighbor, self.n_RB)) + self.sig2  #初始化一个三维数组，用于存储每条 V2V 通信链路的干扰值。数组中填充的是噪声功率（sig2）。

    ######################为环境添加车辆（初始化场景中的车辆）###############################################
        # 添加车：有两个方法：add_new_vehicles(需要传输起始坐标、方向、速度)，add_new_vehicles_by_number（n）。
        # 后者只需要一个参数，n，但是并不是添加n辆车，而是4n辆车，上下左右方向各一台，位置是随机的。

        '''添加有人驾驶汽车'''
        def add_new_HumanDrivenVehicle(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
            self.vehicles.append(HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))
            self.HumanDrivenVehicle_list.append(HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))

        '''添加无人驾驶汽车'''
        def add_new_AutonomousDrivenVehicle(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
            self.vehicles.append(AutonomousDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))
            self.AutonomousDrivenVehicle_list.append(AutonomousDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))

        '''随机添加有人驾驶汽车'''
        def add_new_HumanDrivenVehicle_by_number(self, n):

            for i in range(n):
                agent_id = i
                max_power = 23
                # 随机选择下行车道和起始位置
                ind = np.random.randint(0, len(self.down_lanes))
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'd' # 设定初始方向为向下
                # self.add_new_HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, np.random.randint(10, 15),"Human")# 每辆新车辆的速度在10到15米/秒之间随机。
                self.add_new_HumanDrivenVehicle(agent_id, max_power, start_position, start_direction,
                                                80, "Human")  # 每辆新车辆的速度在10到15米/秒之间随机。

                # 在上行车道添加车辆
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'u'
                self.add_new_HumanDrivenVehicle(agent_id + 1, max_power,start_position, start_direction, 80,"Human")

                # 在左行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
                start_direction = 'l'
                self.add_new_HumanDrivenVehicle(agent_id + 2, max_power,start_position, start_direction, 80,"Human")

                # 在右行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
                start_direction = 'r'
                self.add_new_HumanDrivenVehicle(agent_id + 3, max_power,start_position, start_direction, 80,"Human")

            self.new_channelmodel()
            # # 初始化车辆间通信的信道模型参数
            # self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
            # self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
            #
            # # 计算车辆间的距离变化量
            # self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

        '''随机添加无人驾驶汽车'''
        def add_new_AutonomousDrivenVehicle_by_number(self, n):

            for i in range(n):
                agent_id = 1
                max_power = 100
                # 随机选择下行车道和起始位置
                ind = np.random.randint(0, len(self.down_lanes))
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'd' # 设定初始方向为向下
                self.add_new_AutonomousDrivenVehicle(agent_id, max_power,start_position, start_direction, 80,"Auto")# 每辆新车辆的速度在10到15米/秒之间随机。

                # 在上行车道添加车辆
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'u'
                self.add_new_AutonomousDrivenVehicle(agent_id + 1, max_power,start_position, start_direction, 80,"Auto")

                # 在左行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
                start_direction = 'l'
                self.add_new_AutonomousDrivenVehicle(agent_id + 2, max_power,start_position, start_direction, 80,"Auto")

                # 在右行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
                start_direction = 'r'
                self.add_new_AutonomousDrivenVehicle(agent_id + 3, max_power,start_position, start_direction, 80,"Auto")

            self.new_channelmodel()
        '''初始化车辆之间通信的信道模型参数'''
        def new_channelmodel(self):


        '''计算小尺度衰落信道增益（多普勒）'''
        def apply_doppler_effect(self, h, velocity_A, velocity_B):



    ######################       更新车辆位置          ###########################################
        '''更新车辆位置：遍历每辆车，根据其方向和速度更新位置'''
        def renew_vehicles_positions(self):
            i = 0
            while (i < len(self.vehicles)):
                delta_distance = self.vehicles[i].velocity * self.time_slow         #计算车辆在此时间步长内行驶的距离。车辆速度与时间步长的乘积
                change_direction = False                                                      #车辆在当前更新周期内是否改变方向

                '''如果车辆向上移动（u），检查它是否应该改变方向。'''
                if self.vehicles[i].direction == 'u':
                    #检查左车道十字路口（上行）
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至left（l）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #检查右车道十字路口（上行）
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至right（r）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #如果没有改变则增加垂直位置
                    if change_direction == False:
                        self.vehicles[i].position[1] += delta_distance

                '''如果车辆向下移动（d），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'd') and (change_direction == False):
                    #检查左车道十字路口（下行）
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至left（l）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #检查右车道十字路口（下行）
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至right（r）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则减去垂直位置
                    if change_direction == False:
                        self.vehicles[i].position[1] -= delta_distance

                '''如果车辆向右移动（r），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'r') and (change_direction == False):
                    # 检查左车道十字路口（右行）
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至up（u）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 检查右车道十字路口（右行）
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至down（d）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则继续向右移动，加上去
                    if change_direction == False:
                        self.vehicles[i].position[0] += delta_distance

                '''如果车辆向左移动（l），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'l') and (change_direction == False):
                    # 检查左车道十字路口（左行）
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至up（u）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    # 该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至down（d）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则继续向左移动，减去
                    if change_direction == False:
                            self.vehicles[i].position[0] -= delta_distance

                '''处理退出'''
                #如果超出模拟边界则重新定位
                if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):

                    #改变方向
                    if (self.vehicles[i].direction == 'u'):
                        self.vehicles[i].direction = 'r'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                    else:
                        if (self.vehicles[i].direction == 'd'):
                            self.vehicles[i].direction = 'l'
                            self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                        else:
                            if (self.vehicles[i].direction == 'l'):
                                self.vehicles[i].direction = 'u'
                                self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                            else:
                                if (self.vehicles[i].direction == 'r'):
                                    self.vehicles[i].direction = 'd'
                                    self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

                '''移至列表中的下一辆车'''
                i += 1

    ######################       更新车辆邻居          ###########################################
        '''更新汽车的邻居'''
        def renew_Vehicle_neighbor(self):
            for i in range(len(self.vehicles)):
                self.vehicles[i].neighbors = []     #将第 i 辆车的 neighbors 属性重置为空列表，表示每辆车的邻居需要重新计算。
                self.vehicles[i].actions = []       #将第 i 辆车的 actions 属性重置为空列表，通常表示每辆车的可选动作集需要在之后进行更新。

            #对于每辆车，将其二维位置 (x, y) 转换为一个复数 x + yi，其中 x 对应复数的实部，y 对应虚部。这种表示形式可以简化距离计算。
            z1 = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
            # z2 = np.array([[complex(c.position[0], c.position[1]) for c in self.AutonomousDrivenVehicle]])
            # z = np.concatenate((z1, z2))

            #abs计算复数差值的绝对值，即得到每辆车与其他车之间的欧几里得距离。Distance 是一个二维矩阵，其中 Distance[i][j] 表示第 i 辆车与第 j 辆车之间的距离。
            Distance = abs(z1.T - z1)

            #查找每辆车最近的邻居
            for i in range(len(self.vehicles)):
                sort_idx = np.argsort(Distance[:, i])

                #选择前 n_neighbor 个最近的邻居
                for j in range(self.n_neighbor):
                    self.vehicles[i].neighbors.append(sort_idx[j + 1])

                #将车辆的目的地设置为它的邻居列表
                destination = self.vehicles[i].neighbors
                self.vehicles[i].destinations = destination


    ######################       更新车辆信道          ###########################################
        '''更新车辆信道'''
        def renew_vehicles_channel(self,h, velocity_A, velocity_B):


        '''更新 V2V 信道的快速衰落(是否需要,已修改)'''
        def renew_channels_fastfading(self, h, velocity_A, velocity_B):



        ######################       计算奖励          ###########################################
        '''计算性能奖励（********问题**********）'''
        '''训练阶段，计算 V2I 和 V2V 通信的性能（速率、干扰）和奖励'''

        def Compute_Performance_Reward_Train(self, actions_power):


        def act_for_training(self, actions):
            action_temp = actions.copy()

            # 计算奖励和速率。调用 Compute_Performance_Reward_Train 函数，根据 action_temp 计算
            V2I_SINR_C, V2V_SINR_C, total_rewards, HumanDriven_rewards, Autonomous_rewards, V2I_rewards = self.Compute_Performance_Reward_Train(
                action_temp)

            # 计算 V2V 通信成功率
            V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)

            # 计算总奖励
            lambdda1 = 0.5
            lambdda2 = 1 - lambdda1
            reward = lambdda1 * HumanDriven_rewards.sum() + lambdda2 * Autonomous_rewards.sum()

            V2I_SINR_C = (np.mean(V2I_SINR_C)) / (10 ** 5)
            V2V_SINR_C = (np.mean(V2V_SINR_C)) / (10 ** 6)

            # 返回结果
            return V2I_SINR_C, V2V_SINR_C, total_rewards




    ######################       初始化场景          ###########################################
        '''初始化一个新的随机场景'''
        def new_random_game(self, n_Veh=0):
            #初始化车辆列表
            self.HumanDrivenVehicle_list = []
            self.AutonomousDrivenVehicle_list = []
            self.vehicles = []

            #设置车辆数
            if n_Veh > 0:
                self.n_Veh = n_Veh

            # 根据车辆数量四分之一的比例添加新车辆
            self.add_new_HumanDrivenVehicle_by_number(int(self.n_Veh / 4))
            self.add_new_AutonomousDrivenVehicle_by_number(int(self.n_Veh / 4))


            #更新车辆邻居信息
            self.renew_Vehicle_neighbor()

            #更新车辆信道信息
            self.renew_vehicles_channel(0,0,0)
            self.renew_channels_fastfading(0,0,0)

            #初始化链路需求和时间限制
            self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
            self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
            self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

            #初始化随机基线的链路需求和时间限制
            self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
            self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
            self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')








