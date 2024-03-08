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

pi = np.pi
R2D = 180/pi
D2R = pi/180
tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])

class XB7s(ERobot):
    def __init__(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (50) * mm
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tx(0.03)*ET.tz(0.38)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.34)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tx(0.1095)*ET.tz(0.035)*ET.Rx(), name="link3", parent=l2)
        l4 = Link(ET.tx(0.2255)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tx(0.083)*ET.Ry(pi/2)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(XB7s, self).__init__(elinks, name="XB7s", manufacturer="Rokae")

        self.qr = np.array([0,0,0,0,pi/6,0])
        self.qz = np.zeros(6)
     
class Test_XB7s:
    def __init__(self,robot):
        self.robot = robot
        self.a1 = 0.03
        self.d1 = 0.38
        self.a2 = 0.34
        self.a3 = 0.035
        self.d4 = 0.335
        self.d6 = 0.083
        self.tool_frame =  np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
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
            j= self.ink_a(joint[0],joint[1],joint[2],joint[3],joint[4],joint[5],Px,Py,Pz,tx,ty,tz) # tcp 
            joint_list.append(j)
        joint_list = joint_list[:-1]
        joint_list.append(end_jnt)
        #self.show(x,y,z,joint_list)
        # robot.plot(np.array(joint_list),backend='pyplot',dt=0.1,block=True)
        return joint_list
                 
    def ink_a(self,j1,j2,j3,j4,j5,j6,x,y,z,tx,ty,tz):
        deg = 180/pi
        #print('tcp pose: ',posee)
        cos = np.cos
        sin = np.sin
        T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,self.d1],[0,0,0,1]])
        T12 = np.mat([[sin(j2),cos(j2),0,0],[0,0,1,0],[cos(j2),-sin(j2),0,0],[0,0,0,1]])
        T23 = np.mat([[cos(j3),-sin(j3),0,self.a2],[sin(j3),cos(j3),0,0],[0,0,1,0],[0,0,0,1]])
        T34 = np.mat([[cos(j4),-sin(j4),0,self.a3],[0,0,1,self.d4],[-sin(j4),-cos(j4),0,0],[0,0,0,1]])
        T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,-1,0],[sin(j5),cos(j5),0,0],[0,0,0,1]])
        T56 = np.mat([[-cos(j6),sin(j6),0,0],[0,0,1,self.d6],[sin(j6),cos(j6),0,0],[0,0,0,1]])
        # T6t = np.mat([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
        T02 = np.dot(T01,T12)
        T03 = np.dot(T02,T23)
        T35 = np.dot(T34,T45)
        T36 = np.dot(T35,T56)
        # T3t = np.dot(T36,T6t)
        T3t = T36
        # T06 = np.dot(T03,T36)
        # T0e = np.dot(T06,T6t)
        # print(f"T0e={T0e}")
        # print(f"T36={T36}")
        # print(f"T3t={T3t}")
        # print(f"T36={T36}")
        # 三轴到6轴的偏差
        Px = T3t[0,3]
        Py = T3t[1,3]
        Pz = T3t[2,3]
        # Px = 0.3
        # Py = 0.015
        # Pz = 0.208
        print(f"T3t={Px,Py,Pz}")
        if abs(-Pz/np.sqrt(x**2+y**2)) > 1:
            print(f"计算j1出现不可解的数学问题,{-Pz,np.sqrt(x**2+y**2)}")
            return False
        else:
            c1 = np.arcsin(-Pz/np.sqrt(x**2+y**2))
            b1 = np.arctan2(-y,x)
            theta1 = self.second_sort(c1,b1,j1)
        # theta1 = np.arcsin(-Pz/np.sqrt(x**2+y**2))-np.arctan2(-y,x)
        # print(f"theta1={theta1*deg}")
     
        # get joint2
        m1 = (x+sin(theta1)*Pz-cos(theta1)*self.a1)/cos(theta1)
        m3 = z - self.d1
        n = (m1**2+m3**2+self.a2**2-Px**2-Py**2)/(2*self.a2)
        # print(f"m1={m1},m3={m3},n={n}")
        if abs(n/np.sqrt(m1**2+m3**2)) > 1:
            print(f"计算j2出现不可解的数学问题,{n,np.sqrt(m1**2+m3**2)}")
            return False
        else:
            a2 = np.arcsin(n/np.sqrt(m1**2+m3**2))
            b2 = np.arctan2(m3,m1)
            # print(f"a2={a2},b2={b2}")
            theta2 = self.second_sort(a2,b2,j2)
        # theta2 = np.arcsin(n/np.sqrt(m1**2+m3**2)) - np.arctan2(m3,m1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        # get joint23  
        k = m3 - cos(theta2)*self.a2
        if abs(-k/np.sqrt(Px**2+Py**2)) > 1:
            print(f"计算j3出现不可解的数学问题,{-k,np.sqrt(Px**2+Py**2)}")
            return False
        else:
            a3 = np.arcsin(-k/np.sqrt(Px**2+Py**2))
            b3 = np.arctan2(-Px,Py) 
            # print(f"a3={a3},b3={b3}")
            theta23 = self.second_sort(a3,b3,j2+j3)
        # theta23 = np.arcsin(-k/np.sqrt(Px**2+Py**2)) - np.arctan(-Px/Py) 
        # print(f"theta23={theta23*deg}")
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

    def second_sort(self,alpha,beta,joint):
        # 已知theta = alpha-beta,beta唯一，alpha可能等于pi-alpha
        #print(f"alpha,beta,joint_index={alpha,beta,joint}")
        gamma = np.pi-alpha
        joint1 = alpha-beta
        joint2 = gamma-beta
        # print(f"joint1,joint2={joint1,joint2}")
        # 转化到[-180,180]之间
        joint1 = np.arctan2(np.sin(joint1),np.cos(joint1))
        joint2 = np.arctan2(np.sin(joint2),np.cos(joint2))
        # print(f"joint1,joint2={joint1,joint2}")
        diff1 = abs(joint-joint1)
        diff2 = abs(joint-joint2)
        # print(f"joint1,joint2={joint1*R2D,joint2*R2D}")
        # print(f"diff={diff1},{diff2}")
            
        if diff1<diff2:
            # print(f"joint1={joint1}")
            return joint1
        else:
            # print(f"joint2={joint2}")
            return joint2
      
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
      
def test_XB7s(robot):
    Robot = Test_XB7s(robot)
    start_jnt = np.array([0,0,0,90,90,0])*D2R
    # start_jnt = np.array([82,-18,-152,-99,72,164])*D2R
    # start_jnt = np.array([2.22643,-0.301232,-1.49472,-0.573535,1.0022,-0.110097])
    # start_jnt = np.array([-92,14,25,45,125,72])*D2R
    ans = start_jnt
    print(f"输入的测试角度,取其456轴={ans}")
    fk = robot.fkine(start_jnt)
    # pos = Robot.get_position(SE3(np.dot(fk,tool_frame)))
    pos = Robot.get_position(fk)
    print(f"T0t的位置={pos}")
    j1,j2,j3 = start_jnt[0],start_jnt[1],start_jnt[2]
    j4,j5,j6 =  start_jnt[3],start_jnt[4],start_jnt[5]
    joint = Robot.ink_a(j1,j2,j3,j4,j5,j6,pos[0],pos[1],pos[2],0,0,0.14)
    #print(f"cos5={np.cos(j5)},sin5={np.sin(j5)},cos4={np.cos(j4)}")
    #print(f"p1={p1},p2={p2},l1={l1},l2={l2}")
    print(f"逆解输出的角度={np.array(joint)}")
    
if __name__ == "__main__":
    robot = XB7s()
    test_XB7s(robot)