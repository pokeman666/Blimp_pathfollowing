import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
matplotlib.use('Agg')  # 使用无头模式

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
       
        return self.value

class RGBlimpenv():
    def __init__(self, Time, actionTime,targetpos) -> None:

        self.max_action = np.array([5.0, 5.0], dtype=np.float32)  # 每个维度的最大值
        # 目标位置，通常是一个路径或目标点
        self.targetpos = targetpos  # 修改为 targetpath
        # 成功标志，初始为0
        self.success = 0
        # 完成标志，初始为False
        self.done = False
        # 超出范围标志，初始为False
        self.exceed = False
        
        # 低通滤波
        self.filter_fl = LowPassFilter(0.7)  # 左螺旋桨的低通滤波器
        self.filter_fr = LowPassFilter(0.7)  # 右螺旋桨的低通滤波器
        self.filter_rb0 = LowPassFilter(0.6)  # rb0的低通滤波器

        # Global Parameters
        self.Time = Time
        self.actionTime = actionTime
        self.del_t = 0.01
        self.currentTime = 0.0
        self.Multi = 40

        self.timelength = 1 + int(self.Time / self.actionTime)


        # Simulation Parameters
        # Environment Parameters
        self.Rho = 1.2187  # Kg / m ^ 3 density of Air
        self.rho = 0.1690
        # Mass Parameters
        self.g = 9.8000  # N / kg    gravitational acceleration
        self.m = 0.10482
        self.mb = 0.05407
        self.B = 0.15204
        self.G = 0.06713

        self.Cd0 = 0.2425
        self.Cda = 4.4195
        self.Cdb = 7.5080
        self.Cs0 = 0.0083
        self.Csa = -0.0744
        self.Csb = -2.1140
        self.Cl0 = 0.1594
        self.Cla = 2.9375
        self.Clb = 4.5537

        self.Cmx0 = 0.0131 * 1.00
        self.Cmxa = -0.0301 * 1.00
        self.Cmxb = -0.5256 * 1.00
        self.Cmy0 = 0.0568
        self.Cmya = 0.0933
        self.Cmyb = 5.2357
        self.Cmz0 = 0.0006
        self.Cmza = -0.0012 * 1.00
        self.Cmzb = -0.0936 * 1.00
        self.K1 = -0.0503 * 1.00
        self.K2 = -0.0264 * 1.00
        self.K3 = -0.0137 * 1.00
        # Design Parameters
        # Mass
        self.Ix = 0.0300
        self.Iy = 0.0150
        self.Iz = 0.0100
        self.I = np.diag([self.Ix, self.Iy, self.Iz])

        # Geometry
        self.d = 0.150
        self.A = 0.250
        self.r = np.array([-0.0432, -0.0003, 0.0079])

        # Input
        self.Fl = 0.0  # gf    The output force of left propeller
        self.Fr = 0.0  # gf    The output force of right propeller
        self.rb = np.array([np.random.uniform(0.0247,0.1247), 0.0006, 0.2380])
        # Initial Conditions

        # m     position from the inertial frame origin to the origin of the
        # body-fixed frame
        self.p = np.array([0.0, 0.0, 0.0])
        self.pt=np.array([0.0, 0.0, 0.0])
        self.v = np.array([0.0, 0.0, 0.0])
        self.w = np.array([0.0, 0.0, 0.0])
        self.e = np.array([0.0, 0.0, 0.0])
        self.dist =np.array([0.0,0.0, 0.0])
        self.Rbi = np.eye(3)
        self.Jbi = np.eye(3)


        self.state_dim = 15
        self.action_dim =2

    def reset(self):
        # 重置当前时间为0.0
        self.currentTime = 0.0
        # 重置成功标志为0
        self.success = 0
        # 重置完成标志为False
        self.done = False
        # 重置输入向量为[0., 0., 0.]

        # 随机生成滑块位置
        self.rb = np.array([np.random.uniform(0.0247,0.1247), 0.0006, 0.2380])

        # 初始条件
        # m 从惯性坐标系原点到体固定坐标系原点的位置
        self.p = np.array([0.0, 0.0, 0.0])  # 位置向量
        self.pt = np.array([0.0, 0.0, 0.0])

        #训练时， 目标终点的位置随机生成
        self.targetpos = np.random.uniform(low=[4.0, -2.0, -2.0], high=[5.0, 2.0, 2.0]) 

        # 训练时，初始速度、角速度和欧拉角在一定范围内随机生成
        self.v = np.random.uniform(low=[0.5, 0, -0.2], high=[1.0, 0, 0.2])  # 速度向量
        self.w = np.random.uniform(low=[0, -0.3, 0], high=[0, 0.3, 0])  # 角速度向量
        self.e = np.random.uniform(low=[0, -np.pi / 6, 0], high=[0, np.pi / 6, 0])  # 欧拉角
        # # test时
        # self.v = np.array([0.5,0,0])
        # self.w = np.array([0.0,0.0,0.0])
        # self.e = np.array([0.0,0.0,0.0])

        # 计算位置在xoz平面上的投影
        p_xoz_vector = np.array([self.p[0], 0.0, self.p[2]])
        # 找到目标路径上的最近点
        point = self.findpoint(p_xoz_vector, self.targetpos)
        # 计算距离向量
        self.dist = np.array([point[0] - p_xoz_vector[0], point[2] - p_xoz_vector[2], 0])
        state = np.concatenate((self.v, self.w, self.e, self.p, self.targetpos))

        # 计算旋转矩阵Rbi和雅可比矩阵Jbi
        e = self.e
        self.Rbi = np.array([[np.cos(e[2])*np.cos(e[1]), np.sin(e[2])*np.cos(e[1]), -np.sin(e[1])],
                          [-np.sin(e[2])*np.cos(e[0]) + np.cos(e[2])*np.sin(e[1])*np.sin(e[0]), np.cos(e[2])*np.cos(e[0]) + np.sin(e[2])*np.sin(e[1])*np.sin(e[0]), np.cos(e[1])*np.sin(e[0])],
                          [np.sin(e[2])*np.sin(e[0]) + np.cos(e[2])*np.sin(e[1])*np.cos(e[0]), -np.cos(e[2])*np.sin(e[0]) + np.sin(e[2])*np.sin(e[1])*np.cos(e[0]), np.cos(e[1])*np.cos(e[0])]])
        self.Jbi = np.array([
            [1, np.sin(e[0]) * np.tan(e[1]), np.cos(e[0]) * np.tan(e[1])],
            [0, np.cos(e[0]), -np.sin(e[0])],
            [0, np.sin(e[0]) / np.cos(e[1]), np.cos(e[0]) / np.cos(e[1])]
        ])

        # 计算速度和角速度在惯性坐标系中的表示
        #Rbi是正交矩阵 Rbi.T =  Rbi的逆 求解惯性系下的速度
        vo=self.Rbi.T @ state[0:3]
        #这个函数用于求解Ax = b的问题 求解惯性系下的w
        wo=np.linalg.solve(self.Jbi, state[3:6])
        pt=self.pt
         
        return state,vo,wo,pt,self.p

    # 用于计算向量的叉积
    def skew(self, v):
        A = np.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])
        return A

    def wrap_to_pi(self, theta):
        """将角度包装到[-π, π)区间"""
        wrapped = theta % (2 * np.pi)  # 映射到[0, 2π)
        return wrapped - 2 * np.pi * (wrapped >= np.pi)  # 调整到[-π, π)

    #计算p点在v方向上投影点的距离
    def distance_to_vector(self, point, vector):
        # 计算向量的单位向量，即向量除以其模长
        unit_vector = vector / np.linalg.norm(vector)
        # 计算点在向量方向上的投影，即点与单位向量的点积乘以单位向量
        projection_point = np.dot(point, unit_vector) * unit_vector
        # 计算点与投影点之间的欧几里得距离，即点到向量的垂直距离
        distance = np.linalg.norm(point - projection_point)
        # 返回距离
        return distance
    
    def point_to_line_distance(self, point, direction_vector):
        """
        计算三维空间中点到直线的距离
        假设直线经过原点,沿direction_vector方向延伸
        
        参数：
        point: list/np.array 三维点坐标 [x, y, z]
        direction_vector: list/np.array 三维方向向量 [a, b, c]
        
        返回：
        float 点到直线的距离
        """
        # 转换为numpy数组
        p = np.array(point)
        v = np.array(direction_vector)
        
        # 计算叉乘的模
        cross_product = np.cross(p, v)
        distance = np.linalg.norm(cross_product) / np.linalg.norm(v)
        
        return distance

    def findpoint(self,P, v):

        dot_product = np.dot(P, v)  
        v_length_squared = np.dot(v, v) 
        projection = (dot_product / v_length_squared) * v  
        return projection
    


    def angle_between_vectors(self, v1, v2):
        """
        计算两个向量之间的夹角（以弧度为单位）
        :param v1: 第一个向量，例如 [x1, y1, z1]
        :param v2: 第二个向量，例如 [x2, y2, z2]
        :return: 夹角（弧度）
        """
        # 计算点积
        dot_product = np.dot(v1, v2)
        # 计算向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_v1 * norm_v2)
        # 使用反余弦函数得到夹角（弧度）
        angle = math.acos(cos_theta)
        return angle

    
    def step(self, action):
        self.Fl = self.filter_fl.update(action[0])
        self.Fr = self.filter_fr.update(action[1])

        #update state
        for i in range(int(self.actionTime / self.del_t)):
            Rho = self.Rho  # 空气密度
            A = self.A  # 参考面积

            #这里的速度是体坐标系下的速度
            ##这里的v与self.v指向同一个内存地址
            v = self.v  # 速度
            w = self.w  # 角速度
            p = self.p  # 位置
            e = self.e  # 欧拉角
            # Rotation matrix  惯性系到体坐标系
            Rbi = np.array([[np.cos(e[2])*np.cos(e[1]), np.sin(e[2])*np.cos(e[1]), -np.sin(e[1])],
                          [-np.sin(e[2])*np.cos(e[0]) + np.cos(e[2])*np.sin(e[1])*np.sin(e[0]), np.cos(e[2])*np.cos(e[0]) + np.sin(e[2])*np.sin(e[1])*np.sin(e[0]), np.cos(e[1])*np.sin(e[0])],
                          [np.sin(e[2])*np.sin(e[0]) + np.cos(e[2])*np.sin(e[1])*np.cos(e[0]), -np.cos(e[2])*np.sin(e[0]) + np.sin(e[2])*np.sin(e[1])*np.cos(e[0]), np.cos(e[1])*np.cos(e[0])]])
            V = np.linalg.norm(v)  # m/s
            V2 = np.power(V, 2)

            #攻角和侧滑角都是根据体坐标系下相对于气流的速度计算的
            alpha = np.arctan(v[2] / (v[0] + 1e-20))  # rad, angle of attack
            beta = np.arcsin(v[1] / (V + 1e-20))  # rad, angle of side-slip

            Rbv = np.array([
                [np.cos(alpha) * np.cos(beta), -np.cos(alpha)
                 * np.sin(beta), -np.sin(alpha)],
                [np.sin(beta), np.cos(beta), 0],
                [np.sin(alpha) * np.cos(beta), -np.sin(alpha)
                 * np.sin(beta), np.cos(alpha)]
            ])

            Jbi = np.array([
                [1, np.sin(e[0]) * np.tan(e[1]), np.cos(e[0]) * np.tan(e[1])],
                [0, np.cos(e[0]), -np.sin(e[0])],
                [0, np.sin(e[0]) / np.cos(e[1]), np.cos(e[0]) / np.cos(e[1])]
            ])

            Fl = self.Fl * self.g * 1e-3  # 左舵机控制信号，转换为力
            Fr = self.Fr * self.g * 1e-3  # 右舵机控制信号，转换为力
            rb = self.rb  # rb控制信号
            mb = self.mb  # 质量
            G = self.G  # 重力加速

            # 计算阻力、侧力和升力
            D = 0.5 * Rho * V2 * A * (self.Cd0 + self.Cda * alpha ** 2 + self.Cdb * beta ** 2)  # 阻力
            S = 0.5 * Rho * V2 * A * (self.Cs0 + self.Csa * alpha ** 2 + self.Csb * beta)  # 侧力
            L = 0.5 * Rho * V2 * A * (self.Cl0 + self.Cla * alpha + self.Clb * beta ** 2)  # 升力

            # 计算力矩
            M1 = 0.5 * Rho * V2 * A * (self.Cmx0 + self.Cmxa * alpha + self.Cmxb * beta)  # 滚转力矩
            M2 = 0.5 * Rho * V2 * A * (self.Cmy0 + self.Cmya * alpha + self.Cmyb * beta ** 4)  # 俯仰力矩
            M3 = 0.5 * Rho * V2 * A * (self.Cmz0 + self.Cmza * alpha + self.Cmzb * beta)  # 偏航力矩
            Damping = np.array([self.K1, self.K2, self.K3]) * w    #计算阻尼力矩

            # State Equation
            lg = self.m * self.r + mb * rb
            HF = 1
            ff = ((self.m + mb) * np.cross(v, w) + np.cross(np.cross(w, lg), w) +
                  np.dot(Rbi, np.array([0., 0., G])) +
                  np.dot(Rbv, np.array([-D, S, -L])) +
                         np.array([Fl + Fr, 0., 0.])
                  )
            tt = (np.cross(np.dot((self.I - mb * np.dot(self.skew(rb), self.skew(rb))), w), w) + np.cross(lg, np.cross(v, w)) * HF +
                  np.cross(lg, np.dot(Rbi, np.array([0., 0., self.g]))) + np.dot(Rbv, np.array([M1, M2, M3]) + Damping) +
                  np.array([0., (Fl + Fr) * rb[2], (Fl - Fr) * self.d])
                  )
            #描述系统的惯性特征
            H = np.block([[(self.m + mb) * np.eye(3), -self.skew(lg)],
                          [self.skew(lg), self.I - mb * self.skew(rb) @ self.skew(rb)]])
            #将力与力矩合并成一个向量
            fftt = np.concatenate([ff, tt])
            x_ = np.linalg.solve(H, fftt)
            v_ = x_[0:3]
            w_ = x_[3:6]
            p_ = Rbi.T @ self.v
            p_t = np.array([self.targetpos[0],self.targetpos[1],self.targetpos[2]])
            #求得惯性系下欧拉角变化率
            e_ = np.linalg.solve(Jbi, w)

            # Update
            self.v += v_ * self.del_t
            self.w += w_ * self.del_t
            self.p += p_ * self.del_t
            self.pt = p_t
            self.e += e_ * self.del_t
            self.Rbi = Rbi
            self.Jbi = Jbi

            #位置在xoz平面的投影
            p_xoz_vector = np.array([self.p[0], 0.0, self.p[2]])
            #位置在路径向量上的投影点
            point = self.findpoint(p_xoz_vector, self.targetpos)

            distance =  math.sqrt((point[0]-p_xoz_vector[0])**2 + (point[2]-p_xoz_vector[2])**2)
            #dist中存储了 到路径的距离向量和距离本身
            self.dist = np.array([point[0]-p_xoz_vector[0],point[2]-p_xoz_vector[2],distance])

            # 限制v w e的范围
            v_min = np.array([0.1, -5.0, -1.0])
            v_max = np.array([3.0, 5.0, 1.0])
            w_min = np.array([-10.0, -10.0, -10.0])
            w_max = np.array([10.0, 10.0, 10.0])

            for i in range(3):
                self.e[i] = self.wrap_to_pi(self.e[i])

            self.v = np.clip(self.v, v_min, v_max)
            self.w = np.clip(self.w, w_min, w_max)

            state = np.concatenate((self.v, self.w, self.e, self.p, self.targetpos))
        
        self.currentTime += self.actionTime
        reward = self.reward(state)
        if not self.done:
            if self.currentTime > self.Time - 1e-3:
                self.done = True
            else:
                self.done = False
        else:
            self.done = self.done

        return state, reward, self.done, self.Rbi.T @ state[0:3], np.linalg.solve(self.Jbi, state[3:6]),self.pt,self.Fl,self.Fr,self.rb[0]-0.0747,self.p

    def reward(self, state):
        #求解点到直线的距离 横向偏差
        distance =  self.dist[2]
        lateral_error = self.point_to_line_distance(self.p, self.targetpos)

        #求解航向偏差
        v = self.Rbi.T @ state[0:3]
        target_dir = self.targetpos - self.p
        if np.linalg.norm(target_dir) > 1e-6 and np.linalg.norm(v) > 1e-6:
            target_dir_unit = target_dir / np.linalg.norm(target_dir)
            v_unit = v / np.linalg.norm(v)
            heading_error = np.arccos(np.clip(np.dot(v_unit, target_dir_unit), -1.0, 1.0))#用来惩罚大偏差【0,pi】
            cos_sim = np.dot(v_unit, target_dir_unit)#用来保持小角度 【-1,1】

        #求解进度奖励（沿路径方向的移动）
        dis_to_target = np.linalg.norm(self.p - self.targetpos)
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = dis_to_target
        progress = self.prev_distance - dis_to_target
        self.prev_distance = dis_to_target
        # distance_weight = np.clip(dis_to_target/10.0, 0.1, 1.0)

        # reward = (
        #     - lateral_error ** 2
        #     - heading_error ** 2
        #     + progress * 2.0
        #     + 0.5 * cos_sim  # 方向对齐奖励
        # )
        reward = (
            - lateral_error * 2.0
            - heading_error
            + progress * 2.0
            + 0.5 * cos_sim  # 方向对齐奖励
        )
        if self.p[0] > self.targetpos[0]+0.3:
            self.done = True
        return reward
    
    def path(self,path,P_start_array,PT_array):
        fig = plt.figure(figsize=(16, 4))
        # Plot trajectories on each subplot
        axes3 = plt.subplot(221, projection='3d')
        axes0 = plt.subplot(222)
        axes1 = plt.subplot(223)
        axes2 = plt.subplot(224)
                

        axes0.plot(P_start_array[:int(self.currentTime/self.actionTime)+1, 0],P_start_array[:int(self.currentTime/self.actionTime)+1, 1], 'm-', linewidth=3)
        axes1.plot(P_start_array[:int(self.currentTime/self.actionTime)+1, 0],P_start_array[:int(self.currentTime/self.actionTime)+1, 2], 'm-', linewidth=3)
        axes2.plot(P_start_array[:int(self.currentTime/self.actionTime)+1, 1],P_start_array[:int(self.currentTime/self.actionTime)+1, 2], 'm-', linewidth=3)
        axes3.plot3D(P_start_array[:int(self.currentTime/self.actionTime)+1, 0],P_start_array[:int(self.currentTime/self.actionTime)+1, 1],P_start_array[:int(self.currentTime/self.actionTime)+1, 2],'m-', linewidth=3)


        axes0.plot(PT_array[:2, 0],PT_array[:2, 1], 'b--', linewidth=3)
        axes1.plot(PT_array[:2, 0],PT_array[:2, 2], 'b--', linewidth=3)
        axes2.plot(PT_array[:2, 1],PT_array[:2, 2], 'b--', linewidth=3)
        axes3.plot3D(PT_array[:2, 0],PT_array[:2, 1],PT_array[:2, 2],'b--', linewidth=3)


        max_range = max(
            max(P_start_array[:, 0]) - min(P_start_array[:, 0]),
            max(P_start_array[:, 1]) - min(P_start_array[:, 1]),
            max(P_start_array[:, 2]) - min(P_start_array[:, 2])
        )

        center = (max(P_start_array[:, 0]) +
                  min(P_start_array[:, 0])) / 2
        axes3.set_xlim([center - max_range / 2, center + max_range / 2])

        center = (max(P_start_array[:, 1]) +
                  min(P_start_array[:, 1])) / 2
        axes3.set_ylim([center - max_range / 2, center + max_range / 2])

        center = (max(P_start_array[:, 2]) +
                  min(P_start_array[:, 2])) / 2
        axes3.set_zlim([center - max_range / 2, center + max_range / 2])



        axes3.set_box_aspect([1, 1, 1]) 


        axes0.set_xlabel('X [m]')
        axes0.set_ylabel('Y [m]')
        axes0.set_aspect('equal')
        axes0.set_title('XY Plane')

        axes1.set_xlabel('X [m]')
        axes1.set_ylabel('Z [m]')
        axes1.set_aspect('equal')
        axes1.invert_yaxis()
        axes1.set_title('XZ Plane')

        axes2.set_xlabel('Y [m]')
        axes2.set_ylabel('Z [m]')
        axes2.invert_yaxis()
        axes2.set_aspect('equal')
        axes2.set_title('YZ Plane')

        axes3.set_xlabel('X [m]')  
        axes3.set_ylabel('Y [m]')  
        axes3.set_zlabel('Z [m]')
        axes3.invert_zaxis()
        axes3.set_title('Blimp Trajectory')

       
        plt.savefig(path)
        plt.close()  
        
    def pathall(self,time,V_start_array,W_start_array,E_start_array,P_start_array,Action_array,V_body_array,W_body_array,Dis_array,path):
        fig, axs = plt.subplots(2,4, figsize=(30, 10))
       

        # vb over Time
        axs[0, 0].plot(time[:-1], V_body_array[:, 0],
                        label='vx', color='#FF0000')
        axs[0, 0].plot(time[:-1], V_body_array[:, 1],
                        label='vy', color='#00FF00')
        axs[0, 0].plot(time[:-1], V_body_array[:, 2],
                        label='vz', color='#0000FF')
        axs[0, 0].set_title('vb over Time')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('v')
        axs[0, 0].legend()
        
         # wb over Time
        axs[0, 1].plot(time[:-1], W_body_array[:, 0],
                        label='w-roll', color='#FF0000')
        axs[0, 1].plot(time[:-1], W_body_array[:, 1],
                        label='w-pitch', color='#00FF00')
        axs[0, 1].plot(time[:-1], W_body_array[:, 2],
                        label='w-yaw', color='#0000FF')
        axs[0, 1].set_title('wb over Time')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('w')
        axs[0, 1].legend()
        
        # e over Time
        E_start_array = np.rad2deg(E_start_array)
        axs[0, 2].plot(time[:-1], E_start_array[:, 0],
                        label='roll', color='#FF0000')
        axs[0, 2].plot(time[:-1], E_start_array[:, 1],
                        label='pitch', color='#00FF00')
        axs[0, 2].plot(time[:-1], E_start_array[:, 2],
                        label='yaw', color='#0000FF')
        axs[0, 2].set_title('e over Time')
        axs[0, 2].set_xlabel('Time (s)')
        axs[0, 2].set_ylabel('e')
        axs[0, 2].legend()
        
        # p over Time
        axs[0, 3].plot(time[:-1], P_start_array[:, 0],
                        label='px', color='#FF0000')
        axs[0, 3].plot(time[:-1], P_start_array[:, 1],
                        label='py', color='#00FF00')
        axs[0, 3].plot(time[:-1], P_start_array[:, 2],
                        label='pz', color='#0000FF')
        axs[0 ,3].set_title('p over Time')
        axs[0, 3].set_xlabel('Time (s)')
        axs[0, 3].set_ylabel('p')
        axs[0, 3].legend()
        
         # （v over Time）
        axs[1, 0].plot(time[:-1], V_start_array[:, 0],
                        label='vx', color='#FF0000')
        axs[1, 0].plot(time[:-1], V_start_array[:, 1],
                        label='vy', color='#00FF00')
        axs[1, 0].plot(time[:-1], V_start_array[:, 2],
                        label='vz', color='#0000FF')
        axs[1, 0].plot(time[:-1], np.sqrt(V_start_array[:, 0] * V_start_array[:, 0]+V_start_array[:, 1]
                        * V_start_array[:, 1]+V_start_array[:, 2] * V_start_array[:, 2]), label='v', color='yellow')
        axs[1, 0].set_title('v over Time')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('v')
        axs[1, 0].legend()
        
        # （w over Time）
        # axs[1, 1].plot(time[:-1], W_start_array[:, 0],
        #                 label='wx', color='#FF0000')
        # axs[1, 1].plot(time[:-1], W_start_array[:, 1],
        #                 label='wy', color='#00FF00')
        # axs[1, 1].plot(time[:-1], W_start_array[:, 2],
        #                 label='wz', color='#0000FF')
        # axs[1, 1].set_title('w over Time')
        # axs[1, 1].set_xlabel('Time (s)')
        # axs[1, 1].set_ylabel('w')
        # axs[1, 1].legend()
        axs[1, 1].plot(time[:-1], Dis_array[:,0],
                        label='distance to path x', color='#0000FF')
        # axs[1, 1].plot(time[:-1], Dis_array[:,1],
        #                 label='distance to path z', color='#FF0000')
        # axs[1, 1].plot(time[:-1], Dis_array[:,2],
        #                 label='distance to targetpos xoz', color='#00FF00')
        axs[1, 1].plot(time[:-1], Dis_array[:,1],
                        label='distance to targetpos', color='#FF0000')
        axs[1, 1].plot(time[:-1], Dis_array[:,2],
                        label='distance to path', color='#00FF00')
        # axs[1, 1].plot(time[:-1], np.sqrt(Dis_array[:,0]**2+Dis_array[:,1]**2),
        #                 label='distance to path xoz', color='yellow')
        axs[1, 1].set_title('distances to path over Time')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('distance')
        axs[1, 1].legend()

        
        # input over Time
        axs[1, 2].plot(time[:-2], Action_array[:, 0],
                        label='Fl', color='#FF0000')
        axs[1, 2].plot(time[:-2], Action_array[:, 1],
                        label='Fr', color='#00FF00')
        axs[1, 2].set_ylim(-0.5, 10.5)

        
        ax2 = axs[1,2].twinx()
        ax2.plot(time[:-2], Action_array[:, 2],
                        label='rb0', color='#0000FF')
        ax2.set_ylabel('rb0')
        ax2.set_ylim(-0.06, 0.06)
        
        # 获取两个轴的句柄和标签
        handles1, labels1 = axs[1, 2].get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # 合并句柄和标签并创建综合图例
        handles = handles1 + handles2
        labels = labels1 + labels2
        
        axs[1, 2].set_title('Actions over Time')
        axs[1, 2].set_xlabel('Time (s)')
        axs[1, 2].set_ylabel('Fr Fl')
        axs[1, 2].legend(handles, labels, loc='best')

        # beta over Time
        axs[1, 3].plot(time[:-1], np.arctan(V_body_array[:, 2] /
                        (1e-20+V_body_array[:, 0])), label='alpha', color='#FF0000')
        axs[1, 3].plot(time[:-1], np.arcsin(V_body_array[:, 1]/(1e-20+np.sqrt(V_body_array[:, 0]
                        ** 2+V_body_array[:, 1]**2+V_body_array[:, 2]**2))), label='beta', color='#0000FF')
        axs[1, 3].set_title('alpha and beta over Time')
        axs[1, 3].set_xlabel('Time (s)')
        axs[1, 3].set_ylabel('alpha/beta')
        axs[1, 3].legend()

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
        
        
if __name__ == '__main__':
    Time = 60.0
    actionTime = 0.5
    targetxyz = np.array([60,0,-20])
    kp = 0.05
    ki = 0.0
    kd = 0.0
    env = RGBlimpenv(Time, actionTime, targetxyz,kp,ki,kd)
    env.reset()
    for episode in range(1):
            state,vo,wo,pt,pos= env.reset()
            rewards = 0
            total_success = 0
            avg_reward = 0
            done = False
            count = 0
            Action_array = np.zeros((int(env.Time / env.actionTime), 3))
            V_start_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
            E_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
            W_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
            P_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
            V_body_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
            W_body_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
            # 规定路径
            PT_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
            # 到规定路径的距离
            Dis_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
            
        
            state,vo,wo,pt,pos= env.reset()
            V_start_array[count, :] = vo
            W_start_array[count, :] = wo
            P_start_array[count, :] = pos
            V_body_array[count,:] = state[0:3]
            W_body_array[count,:] = state[3:6]
            E_start_array[count, :] = state[6:9]
            Dis_array[count,:] = state[9:12]
            pos = np.array([pos[0],0,pos[2]])
            Dis_array[count,2] = np.linalg.norm(pos- env.targetpos)
            PT_array[count,:] = pt
            count+=1
        
            while not done:
                
                action = [6.0, 0.0]
                # F = np.random.uniform(0, 20.0) 
                # rb0 = np.random.uniform(-0.05, 0.05) 
                # action = np.array([F,rb0])
                next_state, reward, done, vo_,wo_,pt_,fr,fl,rb0,pos_= env.step(action)

                Action_array[count-1, :] = [fr,fl,rb0]
                V_start_array[count, :] = vo_
                W_start_array[count, :] = wo_
                P_start_array[count, :] = pos_
                V_body_array[count,:] = next_state[0:3]
                W_body_array[count,:] = next_state[3:6]
                E_start_array[count, :] = next_state[6:9]
                Dis_array[count,:] = next_state[9:12]
                pos_ = np.array([pos_[0],0,pos_[2]])
                Dis_array[count,2] = np.linalg.norm(pos_- env.targetpos)
                PT_array[count,:] = pt_
                
                count+=1
                state = next_state
                
            time = np.arange(0, env.Time + 2*env.actionTime, env.actionTime)
            path = f'D:/my_project/FT/aaa_path_4.png'
            env.path(path,P_start_array,PT_array)
            path = f'D:/my_project/FT//aaa_all_plots_4.png'
            env.pathall(time,V_start_array,W_start_array,E_start_array,P_start_array,Action_array,V_body_array,W_body_array,Dis_array,path)
        

        
