import numpy as np
import math
from math import pi
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline as RS
import transforms3d as tfs
import roboticstoolbox as rp
import roboticstoolbox.tools.trajectory as tr           
from roboticstoolbox import xplot
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.Link import Link
from spatialmath import SE3
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH 
from scipy.interpolate import interpn as interpn
from scipy.spatial.transform import Slerp

R2D = 180/pi
D2R = pi/180


### 机器人的模型
class NB4_R580(ERobot):
    def __init__(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (50) * mm
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.333)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.28)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tx(0.3)*ET.tz(0.015)*ET.Rx(), name="link3", parent=l2)
        l4 = Link(ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tx(0.068)*ET.Ry(pi/2)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(NB4_R580, self).__init__(elinks, name="NB4_R580", manufacturer="Rokae")

        self.qr = np.array([0,0,0,0,pi/6,0])
        self.qz = np.zeros(6)
     
class CR7(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.296)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.490)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.360)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.ty(0.150)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.127)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(CR7, self).__init__(elinks, name="CR7", manufacturer="Rokae")  

class SR4(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.355)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.400)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.400)*ET.tx(0.05)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.ty(-0.136)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.1035)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(SR4, self).__init__(elinks, name="SR4", manufacturer="Rokae")  


# 这里将SR4拆分成上下两部分来计算，通过后半部分计算的[px,py,pz]，整体的[x,y,z]，以及前半部分的matrix来重新构建解析解
# 先通过整体计算[x,y,z]，然后后半部分计算[px,py,pz]，再代入计算
class SR4t(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.355)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.400)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        elinks = [l0,l1,l2]
        super(SR4t, self).__init__(elinks, name="SR4t", manufacturer="Rokae") 
    
class SR4b(ERobot):  
    def __init__(self):  
        l3 = Link(ET.tz(0.400)*ET.tx(0.05)*ET.Rz(), name="link3", parent=None)
        l4 = Link(ET.ty(-0.136)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.1035)*ET.Rz(), name="link5", parent=l4)
        elinks = [l3,l4,l5]
        super(SR4b, self).__init__(elinks, name="SR4b", manufacturer="Rokae")  
        
### 机器人的Test
class TestNB4:
    def __init__(self,robot):
        self.robot = robot
        self.d1 = 0.333
        self.d4 = 0.3
        self.d6 = 0.068
        self.a2 = 0.28
        self.a3 = 0.015
        # 6 revolute joint, 6 links
        
    def set_jnt_angle(self,start_angle):
        pose = robot.fkine(start_angle)
        return pose
    
    def get_rotation(self,pose):
        return pose.R
    
    def get_position(self,pose):
        return pose.A[:3,3]
    
    def create_matrix(self, M, N):
        T = np.eye(4,4)
        for i in range(3):
            for j in range(3):
                T[i][j] = M[i][j]
            T[i][3] = N[i]
        return T
        
    def matrix2quaternion(self,RT):
        r = R.from_matrix(RT)
        return r.as_quat()
    
    def get_joint(self,RT):
        # joint = robot.ikine_LM(RT).q
        # joint = robot.ikine_min(RT).q
        joint = robot.ikine_LMS(RT).q
        return joint
    
    def get_joint_list(self,T):
        x,y,z = [],[],[]
        joint_list = []
        # extract trajectory from T
        for i in range(len(T)):
            pos = self.get_position(T[i])
            # rot = self.get_rotation(T[i]) # rotation matrix
            joint = self.get_joint(T[i])
            joint = joint.tolist()
            joint_list.append(joint)
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        return joint_list,x,y,z
        
    def is_equal(self,l1,l2):
        ret = True
        if len(l1) == len(l2):
            for i in range(len(l1)):
                if abs(l1[i]-l2[i]*180/pi)>1e-06:
                    ret = False
        return ret
                    
    def test_ik(self,T0,T1,start_jnt,end_jnt):
        ret = True
        sj = self.get_joint(T0)
        ej = self.get_joint(T1)
        r0 = self.is_equal(sj,start_jnt)
        r1 = self.is_equal(ej,end_jnt)
        if r0 is False or r1 is False:
            ret = False
            print('ik is incorrect!')
            print(r0,sj,start_jnt)
        return ret
        
    def show(self,x,y,z,joint_list):    
        # show trajectory
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(x,y,z,'red') 
        plt.show()   
        # plot joint 
        xplot(np.array(joint_list),block=True)

    def replan_movetype(self,start_jnt,end_jnt,t):
        T0 = self.set_jnt_angle(start_jnt)
        T1 = self.set_jnt_angle(end_jnt)
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint_j = TJ.s.tolist() # 得到456轴
        TL = tr.ctraj(T0,T1,t)
        joint_l,x,y,z =  self.get_joint_list(TL) # 得到xyz
        joint_list = []
        tx,ty,tz = 0,0,0
        tool_frame = np.mat([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
        for i in range(len(TL)):
            joint = joint_j[i]
            Tt = np.dot(TL[i],tool_frame)
            Px = self.get_position(Tt)[0]
            Py = self.get_position(Tt)[1]
            Pz = self.get_position(Tt)[2]
            j= self.ink_a(joint[3],joint[4],joint[5],Px,Py,Pz,tx,ty,tz) # tcp 
            joint_list.append(j)
        joint_list = joint_list[:-1]
        joint_list.append(end_jnt)
        #self.show(x,y,z,joint_list)
        # robot.plot(np.array(joint_list),backend='pyplot',dt=0.1,block=True)
        return joint_list
                 
    def ink_a(self,j4,j5,j6,x,y,z,tx,ty,tz):
        deg = 180/pi
        #print('tcp pose: ',posee)
        a1 = 0
        cos = np.cos
        sin = np.sin
        # print('wrist pose: ', posew)
        # T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,self.d1],[0,0,0,1]])
        # T12 = np.mat([[sin(j2),cos(j2),0,0],[0,0,1,0],[cos(j2),-sin(j2),0,0],[0,0,0,1]])
        # T23 = np.mat([[cos(j3),-sin(j3),0,self.a2],[sin(j3),cos(j3),0,0],[0,0,1,0],[0,0,0,1]])
        T34 = np.mat([[cos(j4),-sin(j4),0,self.a3],[0,0,1,self.d4],[-sin(j4),-cos(j4),0,0],[0,0,0,1]])
        T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,-1,0],[sin(j5),cos(j5),0,0],[0,0,0,1]])
        T56 = np.mat([[-cos(j6),sin(j6),0,0],[0,0,1,self.d6],[sin(j6),cos(j6),0,0],[0,0,0,1]])
        T6t = np.mat([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
        T35 = np.dot(T34,T45)
        T36 = np.dot(T35,T56)
        T3t = np.dot(T36,T6t)
        # print(f"T3t={T3t}")
        # print(f"T36={T36}")
        # 三轴到6轴的偏差
        Px = T3t[0,3]
        Py = T3t[1,3]
        Pz = T3t[2,3]
        theta1 = np.arcsin(-Pz/np.sqrt(x**2+y**2))-np.arctan2(-y,x)
        #print(f"theta1={theta1*deg}")
     
        # get joint2
        m1 = (x+sin(theta1)*Pz-cos(theta1)*a1)/cos(theta1)
        # m2 = (y-cos(theta1)*Pz)/sin(theta1)
        m3 = z - self.d1
        #print(m1,m2)
        n = (m1**2+m3**2+self.a2**2-Px**2-Py**2)/(2*self.a2)
        theta2 = np.arcsin(n/np.sqrt(m1**2+m3**2)) - np.arctan(m3/m1)
        #print(f"theta2={theta2*deg}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        # get joint23  
        k = m3 - cos(theta2)*self.a2
        theta23 = np.arcsin(-k/np.sqrt(Px**2+Py**2)) - np.arctan(-Px/Py) 
        #print(f"theta23={theta23*deg}")
        theta3 = theta23 - theta2
        #print(f"theta3={theta3*deg}")
        # 数值解和解析解对比
        joint = [theta1,theta2,theta3,j4,j5,j6]
        joint1 = [theta1*deg,theta2*deg,theta3*deg,j4*deg,j5*deg,j6*deg]
        #验算
        Tn = robot.fkine(joint)
        jointn = robot.ikine_LM(Tn)
        #jointn = robot.ikine_min(Tn)
        # print(f"新角度={joint1}")
        # print(f"验算角度={jointn.q}")
        posn = self.get_position(Tn)
        # print(f"原位置={x,y,z}")
        # print(f"验算位置={posn}")
        x1,y1,z1 = posn[0],posn[1],posn[2]
        # if (x1-x)>1e-06 or (y1-y)>1e-06 or (z1-z)>1e-06:
        #     print(f"原位置={x,y,z}")
        #     print(f"验算位置={posn}")
        #     print(f"当前角度:{joint1}")
        return joint

    def test_rpy(self,joint_list):
        l = len(joint_list)
        for i in range(0,l-1):
            sm = self.get_rotation(robot.fkine(joint_list[i]))
            em = self.get_rotation(robot.fkine(joint_list[i+1]))
            si = np.linalg.inv(sm)
            rot = np.dot(si,em)
            rot_dis = self.rot_error(sm,em)
            R_quat = R.from_matrix(sm).as_quat()
            # print(R_quat)
            print(rot_dis)
            
    def rot_error(self,r_gt,r_est):
        dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2)) 
        #公式计算结果单位为弧度，转成角度返回
        return dis*180/math.pi

    def rotationMatrixToEulerAngles(self,R) :
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0 
        return np.array([x, y, z])

    def moveL(self,start_jnt,end_jnt,t):
        Ts = self.set_jnt_angle(start_jnt)
        Te = self.set_jnt_angle(end_jnt)
        ps = self.get_position(Ts)
        pe = self.get_position(Te)
        # print(f"起始点={ps}")
        TL = tr.ctraj(Ts,Te,t)
        joint=[]
        for i in range(0,len(TL)):
            p = self.get_position(TL[i])
            j = self.get_joint(TL[i])
            joint.append(j)
            # print(f"第{i}个MoveL路径={p}")
        # print(f"终止点={pe}")
        return joint
    
    def compare_rpy(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        joint_list2 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1,E2 = [],[]
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            jnt2 = joint_list2[i]
            fk1 = robot.fkine(jnt1)
            fk2 = robot.fkine(jnt2)
            R1 = self.get_rotation(fk1)
            R2 = self.get_rotation(fk2)
            eular1 = self.rotationMatrixToEulerAngles(R1)
            eular2 = self.rotationMatrixToEulerAngles(R2)
            # print(f"eular1={eular1}")
            E1.append(eular1)
            E2.append(eular2)
        return E1,E2
    
    def compare_quat(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        Q1 = []
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            # quat = Quaternion(q)
            # ax,angle = quat.to_axis_angle()
            # print(f"angle={angle}")
            Q1.append(q)
        return Q1
           
class TestCR7:
    def __init__(self,robot):
        self.robot = robot
        self.d1 = 0.296
        self.a2 = 0.490
        self.d4 = 0.360
        self.d5 = 0.150
        self.d6 = 0.127
        
    def set_jnt_angle(self,start_angle):
        pose = robot.fkine(start_angle)
        return pose

    def get_rotation(self,pose):
        return pose.R
    
    def get_position(self,pose):
        return pose.A[:3,3]
    
    def create_matrix(self, M, N):
        T = np.eye(4,4)
        for i in range(3):
            for j in range(3):
                T[i][j] = M[i][j]
            T[i][3] = N[i]
        return T
        
    def matrix2quaternion(self,RT):
        r = R.from_matrix(RT)
        return r.as_quat()
    
    def get_joint(self,RT):
        # joint = robot.ikine_LM(RT).q*180/pi
        joint = robot.ikine_min(RT).q*180/pi
        # joint = robot.ikine_LMS(RT).q*180/pi
        return joint
    
    def get_joint_list(self,T):
        x,y,z = [],[],[]
        joint_list = []
        # extract trajectory from T
        for i in range(len(T)):
            pos = self.get_position(T[i])
            # rot = self.get_rotation(T[i]) # rotation matrix
            joint = self.get_joint(T[i])
            joint = joint.tolist()
            joint_list.append(joint)
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        return joint_list,x,y,z
    
    def is_equal(self,l1,l2):
        ret = True
        if len(l1) == len(l2):
            for i in range(len(l1)):
                if abs(l1[i]-l2[i]*180/pi)>1e-06:
                    ret = False
        return ret
        
    def show(self,x,y,z,joint_list):    
        # show trajectory
        # ax1 = plt.axes(projection='3d')
        # ax1.plot3D(x,y,z,'r-*') 
        # plt.show()   
        # plot joint 
        xplot(np.array(joint_list),block=True) 
        jnt = joint_list
        for i in range(len(jnt)):
            for j in range(len(jnt[i])):
                jnt[i][j] = jnt[i][j]*R2D
            print(f"逆解角度={jnt[i]}")
        
    def ink_a(self,joints,x,y,z,tx,ty,tz,joint_index):
        j1 = joint_index[0]
        j2 = joint_index[1]
        j3 = joint_index[2]
        j4 = joints[3]
        j5 = joints[4]
        j6 = joints[5]
        cos = np.cos
        sin = np.sin
        T34 = np.mat([[cos(j4),-sin(j4),0,0],[0,0,-1,-self.d4],[sin(j4),cos(j4),0,0],[0,0,0,1]])
        T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,1,self.d5],[-sin(j5),-cos(j5),0,0],[0,0,0,1]])
        T56 = np.mat([[cos(j6),-sin(j6),0,0],[0,0,-1,-self.d6],[sin(j6),cos(j6),0,0],[0,0,0,1]])
        T35 = np.dot(T34,T45)
        T36 = np.dot(T35,T56)
        Px = T36[0,3]
        Py = T36[1,3]
        Pz = T36[2,3]
        # 计算joint1
        if abs(Pz/np.sqrt(x**2+y**2)) > 1:
            print(f"出现不可解的数学问题,{Pz,np.sqrt(x**2+y**2)}")
            return False
        else:
            a1 = np.arcsin(Pz/np.sqrt(x**2+y**2))
            b1 = np.arctan2(-y,x)
            theta1 = self.second_sort(a1,b1,j1)
        # 计算joint2
        m1 = (x-sin(theta1)*Pz)/cos(theta1)
        m2 = z - self.d1
        n = (Px**2+Py**2-self.a2**2-m1**2-m2**2)/(-2*self.a2)
        if abs(n/np.sqrt(m1**2+m2**2)) > 1:
            print(f"出现不可解的数学问题")
            return False
        else:
            a2 = np.arcsin(n/np.sqrt(m1**2+m2**2))
            b2 = np.arctan2(m2,m1)
            theta2 = self.second_sort(a2,b2,j2)
        # print(f"theta2={theta2*R2D}")
        # 计算joint2-3
        k = m2 - cos(theta2)*self.a2
        if abs(k/np.sqrt(Px**2+Py**2)) > 1:
            print(f"出现不可解的数学问题")
            return False
        else:
            a3 = np.arcsin(k/np.sqrt(Px**2+Py**2))
            b3 = np.arctan2(-Py,Px)
            theta23 =  self.second_sort(a3,b3,j2-j3)
            # print(f"theta2={theta2*R2D},theta23={theta23*R2D}")
            theta3 = theta2-theta23
        # 输出
        joint = [theta1,theta2,theta3,j4,j5,j6]
        jnt = np.array(joint)*R2D
        for i in range(len(jnt)):
            if abs(jnt[i])>175:
                print(f"轴{i+1}超限位")
                print(f"theta1={theta1*R2D},theta2={theta2*R2D},theta23={theta23*R2D},theta3={theta3*R2D}")
        
        if self.judge_position(x,y,z,joint,joints) is True:
            # print(f"joint={jnt}")
            return joint
        else:
            return False
     
    def rotationMatrixToEulerAngles(self,R) :
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0 
        return np.array([x, y, z])

    def rotationMatrixToRotVec(self,R):
        rotvec = R.as_rotvec()
        return rotvec
    
    def replan_movetype(self,start_jnt,end_jnt,t):
        T0 = self.set_jnt_angle(start_jnt)
        T1 = self.set_jnt_angle(end_jnt)
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint_j = TJ.s.tolist() # 得到456轴
        TL = tr.ctraj(T0,T1,t)
        joint_list = []
        xl,yl,zl = [],[],[]
        tx,ty,tz = 0,0,0
        for i in range(0,len(TL)):
            joint = joint_j[i]
            # x = self.get_position(TL[i])[0]
            # y = self.get_position(TL[i])[1]
            # z = self.get_position(TL[i])[2]
            x = self.line(start_jnt,end_jnt,i)[0]
            y = self.line(start_jnt,end_jnt,i)[1]
            z = self.line(start_jnt,end_jnt,i)[2]
            # print(f"第{i+1}条路径MoveL提供的位置{x,y,z}")
            if i<1:
                joint_index = start_jnt
            else:
                joint_index = joint_list[i-1]
            j = self.ink_a(joint,x,y,z,tx,ty,tz,joint_index) # tcp
            joint_list.append(j)
            xl.append(x)
            yl.append(y)
            zl.append(z)
        # joint_list = joint_list[:-1]
        # joint_list.append(end_jnt)
        # robot.plot(np.array(joint_list),backend='pyplot',dt=0.01,block=True)
        # self.show(xl,yl,zl,joint_list)
        return joint_list
       
    def moveL(self,start_jnt,end_jnt,t):
        Ts = self.set_jnt_angle(start_jnt)
        Te = self.set_jnt_angle(end_jnt)
        ps = self.get_position(Ts)
        pe = self.get_position(Te)
        # print(f"起始点={ps}")
        TL = tr.ctraj(Ts,Te,t)
        joint=[]
        for i in range(0,len(TL)):
            p = self.get_position(TL[i])
            j = self.get_joint(TL[i])
            joint.append(j)
            # print(f"第{i}个MoveL路径={p}")
        # print(f"终止点={pe}")
        return joint
        
    def moveAbsJ(self,start_jnt,end_jnt,t):
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint = TJ.s.tolist()
        # robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
        return joint
        
    def test_matrix(self,joint):
        j1 = joint[0]
        j2 = joint[1]
        j3 = joint[2]
        j4 = joint[3]
        j5 = joint[4]
        j6 = joint[5]
        cos = np.cos
        sin = np.sin
        T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,self.d1],[0,0,0,1]])
        T12 = np.mat([[sin(j2),cos(j2),0,0],[0,0,1,0],[cos(j2),-sin(j2),0,0,],[0,0,0,1]])
        T23 = np.mat([[-sin(j3),-cos(j3),0,self.a2],[-cos(j3),sin(j3),0,0],[0,0,-1,0],[0,0,0,1]])
        T34 = np.mat([[cos(j4),-sin(j4),0,0],[0,0,-1,-self.d4],[sin(j4),cos(j4),0,0],[0,0,0,1]])
        T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,1,self.d5],[-sin(j5),-cos(j5),0,0],[0,0,0,1]])
        T56 = np.mat([[cos(j6),-sin(j6),0,0],[0,0,-1,-self.d6],[sin(j6),cos(j6),0,0],[0,0,0,1]])
        # T6t = np.mat([[1,0,0,0],[0,1,0,0],[0,0,1,0,14],[0,0,0,1]])
        T02 = np.dot(T01,T12)
        T03 = np.dot(T02,T23)
        T04 = np.dot(T03,T34)
        T05 = np.dot(T04,T45)
        T06 = np.dot(T05,T56)
        x,y,z = T06[0,3],T06[1,3],T06[2,3]
        T = robot.fkine(joint)
        pos = self.get_position(T)
        print(f"x,y,z={x,y,z}")
        print(f"pos={pos}")
        
        print(f"T06={T06}")
        print(f"fk={T}")
                   
    def second_sort(self,alpha,beta,joint):
        # 已知theta = alpha-beta,beta唯一，alpha可能等于pi-alpha
        #print(f"alpha,beta,joint_index={alpha,beta,joint}")
        gamma = np.pi-alpha
        joint1 = alpha-beta
        joint2 = gamma-beta
        diff1 = abs(joint-joint1)
        diff2 = abs(joint-joint2)
        if abs(joint1)>np.pi:
            joint1 = joint1-np.pi
        if abs(joint2)>np.pi:
            joint2 = joint2-np.pi
            
        if diff1<diff2:
            return joint1
        else:
            return joint2
        
    def judge_position(self,x,y,z,joint,joints):
        Tn = robot.fkine(joint)
        posn = self.get_position(Tn)
        x1,y1,z1 = posn[0],posn[1],posn[2]
        if abs(x1-x)>1e-06 or abs(y1-y)>1e-06 or abs(z1-z)>1e-06:
            # print(f"原位置={x,y,z}")
            # print(f"验算位置={x1,y1,z1}")
            js=joint
            for i in range(len(joint)):
                js[i] = js[i]*R2D
                joints[i] = joints[i]*R2D
            print(f"输入角度:{joints}")
            print(f"新解角度:{js}")
            return False
        else:
            return True
    
    def Offs(self,point,dx,dy,dz):
        point[0] += dx
        point[1] += dy
        point[2] += dz
        return point
    
    def compare_rpy(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        # joint_list2 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotationMatrixToEulerAngles(R1)
            E1.append(eular1)
            # print(f"E1={eular1}")
        E1.pop(0)
        E1.insert(0,E1[0])
        return E1
        
    def compare_quat(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        Q1 = []
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            # quat = Quaternion(q)
            # ax,angle = quat.to_axis_angle()
            # print(f"angle={angle}")
            Q1.append(q)
        return Q1
        
    def line(self,start_jnt,end_jnt,t):
        fk1 = robot.fkine(start_jnt)
        fk2 = robot.fkine(end_jnt)
        p1 = self.get_position(fk1)
        p2 = self.get_position(fk2)
        dis_x = p2[0]-p1[0]
        dis_y = p2[1]-p1[1]
        dis_z = p2[2]-p1[2]   
        pos = np.array([p1[0]+t*dis_x/100,p1[1]+t*dis_y/100,p1[2]+t*dis_z/100])
        return pos
   
    def quat_to_axis_angle(self,quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        angle = 2*np.arccos(w)*R2D
        s = math.sqrt(1-w**2)
        if s < 0.001:
            ax,ay,az = x,y,z
            vec = np.array([ax,ay,az])
        else:
            ax,ay,az = x/s,y/s,z/s
            vec = np.array([ax,ay,az])
        # print(f"angle={angle}")
        return vec,angle
   
    def compare_axis_angle(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        ang,vec = [],[]
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            vector,angle = self.quat_to_axis_angle(q)
            ang.append(angle)
            vec.append(vector)
        return vec,ang
                      
class TestSR4:
    def __init__(self,robot):
        self.robot = robot
        self.d1 = 0.355
        self.dx = 0.4
        self.dy = -0.05
        
    def rotationMatrixToEulerAngles(self,R) :
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0 
        return np.array([x, y, z])

    def rotationMatrixToRotVec(self,R):
        rotvec = R.as_rotvec()
        return rotvec
            
    def set_jnt_angle(self,start_angle):
        pose = self.robot.fkine(start_angle)
        return pose

    def get_rotation(self,pose):
        return pose.R
    
    def get_position(self,pose):
        return pose.A[:3,3]
    
    def create_matrix(self, M, N):
        T = np.eye(4,4)
        for i in range(3):
            for j in range(3):
                T[i][j] = M[i][j]
            T[i][3] = N[i]
        return T
        
    def matrix2quaternion(self,RT):
        r = R.from_matrix(RT)
        return r.as_quat()
    
    def get_joint(self,RT):
        # joint = robot.ikine_LM(RT).q
        # joint = robot.ikine_min(RT).q
        joint = robot.ikine_LMS(RT).q
        return joint
    
    def is_equal(self,l1,l2):
        ret = True
        if len(l1) == len(l2):
            for i in range(len(l1)):
                if abs(l1[i]-l2[i]*180/pi)>1e-06:
                    ret = False
        return ret
                     
    def show(self,x,y,z,joint_list):    
        # show trajectory
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(x,y,z,'r-*') 
        plt.show()   
        # plot joint 
        xplot(np.array(joint_list),block=True) 
        jnt = joint_list
        for i in range(len(jnt)):
            for j in range(len(jnt[i])):
                jnt[i][j] = jnt[i][j]*R2D
            # print(f"逆解角度={jnt[i]}")
       
    def ink_a(self,joints,x,y,z,joint_index):
        j1 = joint_index[0]
        j2 = joint_index[1]
        j3 = joint_index[2]
        j4 = joints[3]
        j5 = joints[4]
        j6 = joints[5]
        cos = np.cos
        sin = np.sin
        # 通过连杆计算T36
        rob_back = SR4b()
        T36 = rob_back.fkine(joints[3:5])
        pose = self.get_position(T36)
        Px,Py,Pz = pose[0],pose[1],pose[2]
        # 计算joint1
        if abs(Py/np.sqrt(x**2+y**2)) > 1:
            print(f"计算j1出现不可解的数学问题,{Py,np.sqrt(x**2+y**2)}")
            return False
        else:
            a1 = np.arcsin(Py/np.sqrt(x**2+y**2))
            b1 = np.arctan2(-y,x)
            theta1 = self.second_sort(a1,b1,j1)
        # print(f"theta1={theta1*R2D}")
        # 计算joint2
        m1 = (x-sin(theta1)*Py)/cos(theta1)
        m2 = z - self.d1
        ind1 = m1*self.dx+m2*self.dy
        ind2 = m2*self.dx-m1*self.dy
        n = (m1**2+m2**2+self.dx**2+self.dy**2-Px**2-Pz**2)/(2)
        if abs(n/np.sqrt(ind1**2+ind2**2)) > 1:
            print(f"计算j2出现不可解的数学问题,{n,np.sqrt(ind1**2+ind2**2)}")
            return False
        else:
            a2 = np.arcsin(n/np.sqrt(ind1**2+ind2**2))
            b2 = np.arctan2(ind2,ind1)
            # print(f"a2={a2*R2D},b2={b2*R2D}")
            theta2 = self.second_sort(a2,b2,j2)
            # print(f"theta2={theta2*R2D}")
        # 计算joint2-3
        k = m2 - cos(theta2)*self.dx-sin(theta2)*self.dy
        if abs(k/np.sqrt(Px**2+Pz**2)) > 1:
            print(f"计算j2-j3出现不可解的数学问题,{k,np.sqrt(Px**2+Pz**2)}")
            return False
        else:
            a3 = np.arcsin(k/np.sqrt(Px**2+Pz**2))
            b3 = np.arctan2(Pz,Px)
            # print(f"a3={a3*R2D},b3={b3*R2D}")
            theta23 =  self.second_sort(a3,b3,j2-j3)
            theta3 = theta2-theta23
            # print(f"j2-j3={(j2-j3)*R2D}")
            # print(f"theta23={theta23*R2D}")
        # 输出
        joint = [theta1,theta2,theta3,j4,j5,j6]
        # jnt = np.array(joint)*R2D
        # for i in range(len(jnt)):
        #     if i == 1:
        #         if abs(jnt[i])>135:
        #             print(f"轴{i+1}超限位")
        #     elif i == 2:
        #         if jnt[i]>140 or jnt[i]<-170:
        #             print(f"轴{i+1}超限位")
        #     elif i == 5:
        #         if abs(jnt[i])>270:
        #             print(f"轴{i+1}超限位")
        #     else:
        #         if abs(jnt[i])>175:
        #             print(f"轴{i+1}超限位")
        
        if self.judge_position(x,y,z,joint,joints) is True:
            # print(f"输出的joint={jnt}")
            return joint
        else:
            # print(f"错误的joint={jnt}")
            eps23 = self.get_another(a3,b3,j2-j3)
            eps3 = theta2-eps23
            joint = [theta1,theta2,eps3,j4,j5,j6]
            # print(f"eps23={eps23*R2D},eps3={eps3*R2D}")
            if self.judge_position(x,y,z,joint,joints) is True:
                return joint
            else:
                return False
       
    def get_base_pos(self,jnt):
        cos,sin = np.cos,np.sin
        j1,j2,j3 = jnt[0],jnt[1],jnt[2]
        x = np.cos(j1)*(np.sin(j2)*self.dx-np.cos(j2)*self.dy)
        y = np.sin(j1)*(np.sin(j2)*self.dx-np.cos(j2)*self.dy)
        z = np.cos(j2)*self.dx+np.sin(j2)*self.dy+self.d1
        T03 = np.mat([[-cos(j1)*cos(j2-j3),sin(j1),cos(j1)*sin(j2-j3),cos(j1)*(sin(j2)*self.dx-cos(j2)*self.dy)],
                  [-sin(j1)*cos(j2-j3),-cos(j1),sin(j1)*sin(j2-j3),sin(j1)*(sin(j2)*self.dx-cos(j2)*self.dy)],
                  [sin(j2-j3),0,cos(j2-j3),cos(j2)*self.dx+sin(j2)*self.dy+self.d1],
                  [0,0,0,1]])
        return x,y,z,T03
       
    def replan_movetype(self,start_jnt,end_jnt,t):
        T0 = self.set_jnt_angle(start_jnt)
        T1 = self.set_jnt_angle(end_jnt)
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint_j = TJ.s.tolist() # 得到456轴
        TL = tr.ctraj(T0,T1,t)
        joint_list = []
        xl,yl,zl = [],[],[]
        for i in range(0,len(TL)):
            joint = joint_j[i]
            x = self.get_position(TL[i])[0]
            y = self.get_position(TL[i])[1]
            z = self.get_position(TL[i])[2]
            ## 暂且使用这种直线插补
            # x = self.line(start_jnt,end_jnt,i)[0]
            # y = self.line(start_jnt,end_jnt,i)[1]
            # z = self.line(start_jnt,end_jnt,i)[2]
            # print(f"第{i+1}条路径MoveL提供的位置{x,y,z}")
            if i<1:
                joint_index = start_jnt
            else:
                joint_index = joint_list[i-1]
            j = self.ink_a(joint,x,y,z,joint_index) # tcp
            joint_list.append(j)
            xl.append(x)
            yl.append(y)
            zl.append(z)
        # joint_list = joint_list[:-1]
        # joint_list.append(end_jnt)
        # robot.plot(np.array(joint_list),backend='pyplot',dt=0.01,block=True)
        # self.show(xl,yl,zl,joint_list)
        return joint_list
     
    def line(self,start_jnt,end_jnt,t):
        fk1 = robot.fkine(start_jnt)
        fk2 = robot.fkine(end_jnt)
        p1 = self.get_position(fk1)
        p2 = self.get_position(fk2)
        dis_x = p2[0]-p1[0]
        dis_y = p2[1]-p1[1]
        dis_z = p2[2]-p1[2]   
        pos = np.array([p1[0]+t*dis_x/100,p1[1]+t*dis_y/100,p1[2]+t*dis_z/100])
        return pos
    
    def second_sort(self,alpha,beta,joint):
        # 已知theta = alpha-beta,beta唯一，alpha可能等于pi-alpha
        #print(f"alpha,beta,joint_index={alpha,beta,joint}")
        gamma = np.pi-alpha
        joint1 = alpha-beta
        joint2 = gamma-beta
        # print(f"joint1,joint2={joint1*R2D,joint2*R2D}")
        # 转化到[-180,180]之间
        joint1 = np.arctan2(np.sin(joint1),np.cos(joint1))
        joint2 = np.arctan2(np.sin(joint2),np.cos(joint2))
        diff1 = abs(joint-joint1)
        diff2 = abs(joint-joint2)
        # print(f"joint1,joint2={joint1*R2D,joint2*R2D}")
        # print(f"diff={diff1},{diff2}")
            
        if diff1<diff2:
            return joint1
        else:
            return joint2
        
    def judge_position(self,x,y,z,joint,joints):
        Tn = robot.fkine(joint)
        posn = self.get_position(Tn)
        x1,y1,z1 = posn[0],posn[1],posn[2]
        if abs(x1-x)>1e-06 or abs(y1-y)>1e-06 or abs(z1-z)>1e-06:
            # print(f"原位置={x,y,z}")
            # print(f"验算位置={x1,y1,z1}")
            js=joint
            for i in range(len(joint)):
                js[i] = js[i]*R2D
                joints[i] = joints[i]*R2D
            # print(f"输入角度:{joints}")
            # print(f"新解角度:{js}")
            return False
        else:
            return True
    
    def moveAbsJ(self,start_jnt,end_jnt,t):
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint = TJ.s.tolist()
        # robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
        return joint
        
    def get_another(self,alpha,beta,joint):
        gamma = np.pi-alpha
        joint1 = alpha-beta
        joint2 = gamma-beta
        joint1 = np.arctan2(np.sin(joint1),np.cos(joint1))
        joint2 = np.arctan2(np.sin(joint2),np.cos(joint2))
        diff1 = abs(joint-joint1)
        diff2 = abs(joint-joint2)
        if diff1>diff2:
            return joint1
        else:
            return joint2
    
    def compare_quat(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        Q1 = []
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            # quat = Quaternion(q)
            # ax,angle = quat.to_axis_angle()
            # print(f"angle={angle}")
            Q1.append(q)
        return Q1
             
    def compare_axis_angle(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        ang,vec = [],[]
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            vector,angle = self.quat_to_axis_angle(q)
            ang.append(angle)
            vec.append(vector)
        return vec,ang
    
    def quat_to_axis_angle(self,quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        angle = 2*np.arccos(w)*R2D
        s = math.sqrt(1-w**2)
        if s < 0.001:
            ax,ay,az = x,y,z
            vec = np.array([ax,ay,az])
        else:
            ax,ay,az = x/s,y/s,z/s
            vec = np.array([ax,ay,az])
        # print(f"angle={angle}")
        return vec,angle
   
    def compare_rpy(self,start_jnt,end_jnt,t):
        joint_list1 = self.replan_movetype(start_jnt,end_jnt,t)
        # joint_list2 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotationMatrixToEulerAngles(R1)
            E1.append(eular1)
            # print(f"E1={eular1}")
        E1.pop(0)
        E1.insert(0,E1[0])
        return E1
    
    def comparel_rpy(self,start_jnt,end_jnt,t):
        joint_list1 = self.moveL(start_jnt,end_jnt,t)
        # joint_list2 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotationMatrixToEulerAngles(R1)
            E1.append(eular1)
            # print(f"E1={eular1}")
        E1.pop(0)
        E1.insert(0,E1[0])
        return E1
    
    def compare_movel_angle(self,start_jnt,end_jnt,t):
        joint_list1 = self.moveL(start_jnt,end_jnt,t)
        ang,vec = [],[]
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            vector,angle = self.quat_to_axis_angle(q)
            ang.append(angle)
            vec.append(vector)
        return vec,ang
    
    def compare_moveJ_angle(self,start_jnt,end_jnt,t):
        joint_list1 = self.moveAbsJ(start_jnt,end_jnt,t)
        ang,vec = [],[]
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            q = self.matrix2quaternion(R1)
            vector,angle = self.quat_to_axis_angle(q)
            ang.append(angle)
            vec.append(vector)
        return vec,ang
    
    def moveL(self,start_jnt,end_jnt,t):
        Ts = self.set_jnt_angle(start_jnt)
        Te = self.set_jnt_angle(end_jnt)
        # print(f"起始点={ps}")
        TL = tr.ctraj(Ts,Te,t)
        joint=[]
        for i in range(0,len(TL)):
            p = self.get_position(TL[i])
            j = self.get_joint(TL[i])
            joint.append(j)
            # print(f"第{i}个MoveL路径={p}")
        # print(f"终止点={pe}")
        return joint
            
        
### 独立功能的test
def test_CR7(robot):
    t = np.linspace(0,1,50) # time from 0 to 50s, time interval is 1s
    Robot = TestCR7(robot)
    start_jnt = np.array([-14.357964404225445, -28.140005916481435, 72.92610750878566, -38.11734805316985, 77.26327365781283, -99.07761233005768])
    end_jnt = np.array([-17.798752674312166, 44.47673029618562, 17.46216425744234, -38.11734805316985, 77.26327365781283, -99.07761233005768])
    print(f"测试角度1={start_jnt},测试角度2={end_jnt}")
    robot.plot(np.array(start_jnt),backend='pyplot',dt=1,block=True)
    fk1 = robot.fkine(start_jnt*D2R)
    pos1 = Robot.get_position(fk1)
    fk2 = robot.fkine(end_jnt*D2R)
    pos2 = Robot.get_position(fk2)
    print(f"测试位置1={pos1},测试位置2={pos2}")
    
def test_SR4(robot):
    Robot = TestSR4(robot)
    start_jnt = np.array([0,0,90,0,90,0])*D2R
    end_jnt = np.array([0,2.5469143534933663,5.041841615750075,0, -43.78905655,15.17299192])*D2R #[0.0686628  0.136      1.22759012]
    print(f"测试角度1={start_jnt},测试角度2={end_jnt}")
    # robot.plot(np.array(start_jnt),backend='pyplot',dt=1,block=True)
    # robot.plot(np.array(end_jnt),backend='pyplot',dt=1,block=True)
    fk1 = robot.fkine(start_jnt)
    pos1 = Robot.get_position(fk1)
    fk2 = robot.fkine(end_jnt)
    pos2 = Robot.get_position(fk2)
    print(f"测试位置1={pos1},测试位置2={pos2}")

def test_NB4(robot):
    t = np.linspace(0,1,115)
    ts = np.linspace(0,1,230)
    Robot = TestNB4(robot)
    j1 = np.array([60,0,0,-30,45,0])*D2R
    j2 = np.array([-45,0,15,-30,-45,0])*D2R

    # joint_list = Robot.replan_movetype(j1,j2,t)
    quat1 = Robot.compare_quat(j1,j2,t)
    quat2 = Robot.compare_quat(j2,j1,t)
    quat = quat1+quat2
    plt.subplot(2,1,1)
    plt.title("Rokae Wrist Singular Path Quat")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(ts,quat)
    
    quat_init = quat[0]
    quat_diff = []
    for i in range(230):
        x = abs(quat_init[0]-quat[i][0])
        y = abs(quat_init[1]-quat[i][1])
        z = abs(quat_init[2]-quat[i][2])
        w = abs(quat_init[3]-quat[i][3])
        quad = np.array([x,y,z,w])
        quat_diff.append(quad)
    
    plt.subplot(2,1,2)
    plt.title("Rokae Wrist Singular Path Quat Difference")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(ts,quat_diff)
    plt.show()
    
def test_CR7mat(robot):
    Robot = TestCR7(robot)
    ### CR7
    for i in range(6):
        print(f"第{i}次测试")
        if i == 0:
            start_jnt = np.array([0,0,0,0,0,0])
        elif i == 1:
            start_jnt = np.array([16.799*D2R,0.0,5.897*D2R,-31.101*D2R,-12.182*D2R,17.623*D2R]) # CR7 xyz=[-58.431,-166.263,1261.975]
        elif i == 2:
            start_jnt = np.array([71.034*D2R,-7.232*D2R,104.896*D2R,78.630*D2R,-15.294*D2R,23.510*D2R]) #xyz=[-187.233,-534.709,742.700]
        elif i == 3:
            start_jnt = np.array([0,0,45,90,90,0])*D2R
        elif i == 4:
            start_jnt = np.array([82,-18,152,-99,72,164])*D2R
        else:
            start_jnt = np.array([66.671*D2R,1.863*D2R,124.05*D2R,0.0,-3.726*D2R,27.803*D2R]) #xyz=[-20.821,-427.063,533.455]
            
        #robot.plot(np.array(start_jnt),backend='pyplot',dt=1,block=True)
        Robot.test_matrix(start_jnt)

def test_singular(robot):
    t = np.linspace(0,1,100) # time from 0 to 31s, time interval is 1s
    # Robot = TestCR7(robot)
    Robot = TestSR4(robot)
    # set start point & end point
    j1 = np.array([101,24,-52,29,60,30])*D2R
    j2 = np.array([-124,-63,35,24,-45,15])*D2R
    js = j1*R2D
    je = j2*R2D
    print(f"起始点角度={js}")
    print(f"终止点角度={je}")
    joint_list = Robot.replan_movetype(j1,j2,t)
    # 计算last point,看是否到达了
    # fk1 = robot.fkine(j1)
    # fk2 = robot.fkine(j2)
    # fk3 = robot.fkine(joint_list[-1])
    # spos = Robot.get_position(fk1)
    # epos = Robot.get_position(fk2)
    # pos1 = Robot.get_position(fk3)
    # if abs(pos1[0]-epos[0])>1e-06 or abs(pos1[1]-epos[1])>1e-06 or abs(pos1[2]-epos[2])>1e-06:
    #     print(f"初始位置={spos},终点位置={epos},终止位置={pos1}")
    # pass
    
def test_traj(robot):
    t = np.linspace(0,1,100) # time from 0 to 31s, time interval is 1s
    Robot = TestCR7(robot)
    j1 = np.array([0,0,-90,0,0,0])*D2R
    j2 = np.array([-37.842,13.371,-95.637,-66.976,41.384,60.475])*D2R
    j3 = np.array([-31.314,18.896,-114.276,-41.705,51.453,29.043])*D2R
    j4 = np.array([-0.016,6.841,-116.277,-0.111,33.088,0.111])*D2R
    Ts = robot.fkine(j2)
    Te = robot.fkine(j3)
    TL = tr.ctraj(Ts,Te,t)
    # robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
     
def test_rpy(robot):
    t = np.linspace(0,1,100) # time from 0 to 31s, time interval is 1s
    ts = np.linspace(0,1,400)
    ### CR7对比
    # Robot = TestCR7(robot)
    # # j1 = np.array([0,0,-90,0,0,0])*D2R
    # # j2 = np.array([-37.842,13.371,-95.637,-66.976,41.384,60.475])*D2R
    # # j3 = np.array([-31.314,18.896,-114.276,-41.705,51.453,29.043])*D2R
    # # j4 = np.array([-0.016,6.841,-116.277,-0.111,33.088,0.111])*D2R
    # j1 = np.array([0,30,-90,0,-45,0])*D2R
    # j2 = np.array([12.707,17.913,-112.456,-5.667,-35.115,16.926])*D2R
    # j3 = np.array([-0.002,10.181,-121.185,0.010,-33.634,-0.011])*D2R
    # j4 = np.array([-11.997,24.206,-98.148,4.487,-43.041,-14.876])*D2R
    
    ### SR4对比
    Robot = TestSR4(robot)
    j1 = np.array([-1.97868,0.0102006,-2.25514,-5.08054,-0.861545,5.96648])
    j2 = np.array([-1.91325,-0.033772,-2.20681,-5.19185,-0.742167,6.21637])
    # here,2cm
    j3 = np.array([-1.23933,-0.2999,-2.59575,-4.46444,0.136083,6.04803])
    j4 = np.array([-1.00695,-0.215597,-2.4336,-4.14423,0.471788,5.81443])
    j5 = np.array([-0.968725,-0.157916,-2.34316,-4.11989,0.55805,5.75134])  
    # 对比rpy
    e1 = Robot.compare_rpy(j1,j2,t)
    e2 = Robot.compare_rpy(j2,j3,t)
    e3 = Robot.compare_rpy(j3,j4,t)
    e4 = Robot.compare_rpy(j4,j5,t)
    es = e1+e2+e3+e4
    e,ei=[],[]
    for i in range(len(es)):
        es[i] = es[i]*R2D

    ea = Robot.compare_rpy(j1,j2,t)
    eb = Robot.compare_rpy(j2,j3,t)
    ec = Robot.compare_rpy(j3,j4,t)
    ed = Robot.compare_rpy(j4,j5,t)
    el = ea+eb+ec+ed
    for i in range(len(es)):
        el[i] = el[i]*R2D

    # plt.subplot(2,1,1)
    # plt.title("SR4 Wrist Singular Path RPY")
    # plt.plot(ts,es)
    # plt.ylabel("deg")
    # plt.legend(['r','p','y'],loc="upper left")
    # # plt.plot(t,ei)
    # # plt.legend(['r_init','p_init','y_init'],loc="lower left")
    
    # plt.subplot(2,1,2)
    # plt.title("SR4 MoveL Only Path RPY")
    # plt.plot(ts,el)
    # plt.legend(['r','p','y'],loc="upper left")
    # # plt.legend(['r','p','y'])
    # plt.show()
    plt.title("SR4 Path RPY")
    plt.plot(ts,es,linestyle='-')
    plt.legend(['MoveH r,','MoveH p','MoveH y'],loc="upper left")
    plt.plot(ts,el,linestyle='-.')
    plt.legend(['MoveL r,','MoveL p','MoveL y'],loc="upper right")
    plt.ylabel("deg")
    plt.show()
    
def test_quat(robot):    
    t = np.linspace(0,1,100) # time from 0 to 31s, time interval is 1s
    Robot = TestCR7(robot)
    j1 = np.array([0,0,-90,0,0,0])*D2R
    j2 = np.array([-37.842,13.371,-95.637,-66.976,41.384,60.475])*D2R
    j3 = np.array([-31.314,18.896,-114.276,-41.705,51.453,29.043])*D2R
    j4 = np.array([-0.016,6.841,-116.277,-0.111,33.088,0.111])*D2R
    q1 = Robot.compare_quat(j1,j2,t)
    q2 = Robot.compare_quat(j2,j3,t)
    q3 = Robot.compare_quat(j3,j4,t)
    q4 = Robot.compare_quat(j4,j1,t)
    q = q1+q2+q3+q4
    ts = np.linspace(0,1,400)
    plt.subplot(2,2,1)
    plt.title("Path1 MoveH Quat")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(t,q1)
    plt.subplot(2,2,2)
    plt.title("Path2 MoveH Quat")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(t,q2)
    plt.subplot(2,2,3)
    plt.title("Path3 MoveH Quat")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(t,q3)
    plt.subplot(2,2,4)
    plt.title("Path4 MoveH Quat")
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.plot(t,q4)
    plt.show()
    
    plt.title("MoveH Quat")
    plt.plot(ts,q)
    plt.legend(['x','y','z','w'],loc="upper left")
    plt.show()
    
def test_axis_angle(robot):
    t = np.linspace(0,1,100) # time from 0 to 31s, time interval is 1s
    ### CR7测试
    # Robot = TestCR7(robot)
    ## 电视加工路径
    # j1 = np.array([0,0,-90,0,0,0])*D2R
    # j2 = np.array([-37.842,13.371,-95.637,-66.976,41.384,60.475])*D2R
    # j3 = np.array([-31.314,18.896,-114.276,-41.705,51.453,29.043])*D2R
    # j4 = np.array([-0.016,6.841,-116.277,-0.111,33.088,0.111])*D2R

    
    ### SR4测试
    Robot = TestSR4(robot)
    # j1 = np.array([0,0,90,0,90,0])*D2R
    # j2 = np.array([12.959,-11.494,78.784,0,89.722,12.959])*D2R
    # j3 = np.array([0,-28.614,57.830,0,93.556,0])*D2R
    # j4 = np.array([-12.150,-19.851,69.177,0,90.972,-12.150])*D2R
    ## 
    j1 = np.array([-1.97868,0.0102006,-2.25514,-5.08054,-0.861545,5.96648])
    j2 = np.array([-1.91325,-0.033772,-2.20681,-5.19185,-0.742167,6.21637])
    # here,2cm
    j3 = np.array([-1.00695,-0.215597,-2.4336,-4.14423,0.471788,5.81443])
    j4 = np.array([-1.18568,-0.170659,-2.13529,-3.29044,0.41859,4.71983])
    j5 = np.array([-1.6205,-0.0843001,-1.96096,-2.23665,0.663501,3.5889])  
    ### 机器人运动展示
    jnt_lst1 = Robot.replan_movetype(j1,j2,t)
    jnt_lst2 = Robot.replan_movetype(j2,j3,t)
    jnt_lst3 = Robot.replan_movetype(j3,j4,t)
    jnt_lst4 = Robot.replan_movetype(j4,j5,t)
    jnt_lst = jnt_lst1+jnt_lst2+jnt_lst3+jnt_lst4
    # robot.plot(np.array(jnt_lst),backend='pyplot',dt=0.1,block=True)
    ### 对轴角的判定
    v1,a1 = Robot.compare_axis_angle(j1,j2,t)  
    big = max(a1)
    sml = min(a1)
    init = a1[0]
    dif1 = abs(big-init)
    dif2 = abs(sml-init)
    if dif1>dif2:
        dif = dif1
    else:
        dif = dif2
    print(f"Path1最大值={big},最小值={sml},轴角差值={dif}")
    v2,a2 = Robot.compare_axis_angle(j2,j3,t)
    big = max(a2)
    sml = min(a2)
    init = a2[0]
    dif1 = abs(big-init)
    dif2 = abs(sml-init)
    if dif1>dif2:
        dif = dif1
    else:
        dif = dif2
    print(f"Path2最大值={big},最小值={sml},轴角差值={dif}")
    v3,a3 = Robot.compare_axis_angle(j3,j4,t)
    big = max(a3)
    sml = min(a3)
    init = a3[0]
    dif1 = abs(big-init)
    dif2 = abs(sml-init)
    if dif1>dif2:
        dif = dif1
    else:
        dif = dif2
    print(f"Path3最大值={big},最小值={sml},轴角差值={dif}")
    v4,a4 = Robot.compare_axis_angle(j4,j1,t)
    big = max(a4)
    sml = min(a4)
    init = a4[0]
    dif1 = abs(big-init)
    dif2 = abs(sml-init)
    if dif1>dif2:
        dif = dif1
    else:
        dif = dif2
    print(f"Path4最大值={big},最小值={sml},轴角差值={dif}")
    ts = np.linspace(0,1,400)
    aa = a1+a2+a3+a4
    # plt.subplot(2,2,1)
    # plt.title("Path1 MoveH Axis-Angle")
    # plt.plot(t,a1)
    # plt.ylabel("deg")
    # plt.ylim(220, 250)
    
    # plt.subplot(2,2,2)
    # plt.title("Path2 MoveH Axis-Angle")
    # plt.plot(t,a2)
    # plt.ylabel("deg")
    # plt.ylim(220, 250)
    
    # plt.subplot(2,2,3)
    # plt.title("Path3 MoveH Axis-Angle")
    # plt.plot(t,a3)
    # plt.ylabel("deg")
    # plt.ylim(220, 250)
    
    # plt.subplot(2,2,4)
    # plt.title("Path4 MoveH Axis-Angle")
    # plt.plot(t,a4)
    # plt.ylabel("deg")
    # plt.ylim(220, 250)
    # plt.show()
    
    plt.title("SR4 Path Axis-Angle Difference")
    plt.plot(ts,aa)
    plt.ylabel("deg")
    plt.show()
   
def testLink():
    robo1 = SR4t()
    robo2 = SR4b()
    robo = SR4()
    Robot = TestSR4(robo)
    Robo1 = TestSR4(robo1)
    Robo2 = TestSR4(robo2)
    # jnt = np.array([24,-52,14,103,-5,11])*D2R
    jnt = np.array([0,0,0,0,-90,30])*D2R
    jntt = jnt[0:2]
    jntb = jnt[3:5]
    fk = robo.fkine(jnt)
    fkt = robo1.fkine(jntt)
    fkb = robo2.fkine(jntb)
    fkf = np.dot(fkt,fkb)
    # pz = Robot.get_position(fkb)[2]
    # x,y = Robot.get_position(fk)[0],Robot.get_position(fk)[1]
    # print(f"param={pz,x,y,np.sqrt(x**2+y**2)}")
    # print(f"pos={pos1},{pos2}")
    j1,j2,j3 = jnt[0],jnt[1],jnt[2]
    cos = np.cos
    sin = np.sin
    dx = 0.4
    dy = -0.05
    d1 = 0.355
    R03 = np.mat([[cos(j1)*sin(j2-j3),-cos(j1)*cos(j2-j3),sin(j1),cos(j1)*(sin(j2)*dx-cos(j2)*dy)],
                  [sin(j1)*sin(j2-j3),-sin(j1)*cos(j2-j3),-cos(j1),sin(j1)*(sin(j2)*dx-cos(j2)*dy)],
                  [cos(j2-j3),sin(j2-j3),0,cos(j2)*dx+sin(j2)*dy+d1],
                  [0,0,0,1]])
    T03 = np.mat([[-cos(j1)*cos(j2-j3),sin(j1),cos(j1)*sin(j2-j3),cos(j1)*(sin(j2)*dx-cos(j2)*dy)],
                  [-sin(j1)*cos(j2-j3),-cos(j1),sin(j1)*sin(j2-j3),sin(j1)*(sin(j2)*dx-cos(j2)*dy)],
                  [sin(j2-j3),0,cos(j2-j3),cos(j2)*dx+sin(j2)*dy+d1],
                  [0,0,0,1]])
    T06 = np.dot(T03,fkb)
    print(f"R03={R03}")
    print(f"T03={T03}")
    print(f"前半段连杆={fkt}")
    print(f"T06={T06}")
    print(f"全部连杆={fk}")
    
def test_group_points(robot):
    t = np.linspace(0,1,100)
    ts = np.linspace(0,1,400)
    Robot = TestSR4(robot)
    j1 = np.array([2.29462,-0.0567318,1.23281,1.69754,-1.69123,2.02697])
    j2 = np.array([1.90707,-0.0951564,1.40516,1.92043,-1.1184,2.1223])
    j3 = np.array([1.71149,0.0191068,1.67664,1.97322,-0.808498,2.43125])
    j4 = np.array([1.35017,0.00635404,1.71835,2.49319,-0.536207,1.95816])
    j5 = np.array([1.11819,-0.104611,1.77872,3.19062,-0.460965,1.31929]) 

    v1,a1 = Robot.compare_axis_angle(j1,j2,t)
    v2,a2 = Robot.compare_axis_angle(j2,j3,t)
    v3,a3 = Robot.compare_axis_angle(j3,j4,t)
    v4,a4 = Robot.compare_axis_angle(j4,j5,t)
    a = [a1,a2,a3,a4]
    
    jnt1 = Robot.replan_movetype(j1,j2,t)
    jnt2 = Robot.replan_movetype(j2,j3,t)
    jnt3 = Robot.replan_movetype(j3,j4,t)
    jnt4 = Robot.replan_movetype(j4,j5,t)
    jnt_lst = jnt1+jnt2+jnt3+jnt4
    robot.plot(np.array(jnt_lst),backend='pyplot',dt=0.1,block=True)
    
    x1,b1 = Robot.compare_movel_angle(j1,j2,t)
    x2,b2 = Robot.compare_movel_angle(j2,j3,t)
    x3,b3 = Robot.compare_movel_angle(j3,j4,t)
    x4,b4 = Robot.compare_movel_angle(j4,j5,t)
    b = [b1,b2,b3,b4]
    
    joint1 = Robot.moveAbsJ(j1,j2,t)
    joint2 = Robot.moveAbsJ(j2,j3,t)
    joint3 = Robot.moveAbsJ(j3,j4,t)
    joint4 = Robot.moveAbsJ(j4,j5,t)
    joints = joint1+joint2+joint3+joint4
    xplot(np.array(joints),block=True) 
    robot.plot(np.array(joints),backend='pyplot',dt=0.1,block=True)
    
    diff=[]
    for i in range(len(a)):
        for j in range(len(a[i])):
            dif = abs(a[i][j]-b[i][j])
            diff.append(dif)

    plt.title("Path Axis Angle Difference")
    plt.plot(ts,diff)
    plt.ylabel("deg")
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    robot = CR7()
    # robot = NB4_R580()
    # robot = SR4()
    # testLink()
    
    test_CR7mat(robot)

    
    
