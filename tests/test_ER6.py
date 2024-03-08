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
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

R2D = 180/pi
D2R = pi/180
cos = np.cos
sin = np.sin
tool_frame = np.mat([[1,0,0,0.05],[0,1,0,0.1],[0,0,1,0.14],[0,0,0,1]])

class ER6(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.404)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.4375)*ET.Ry(), name="link2", parent=l1)
        # l2 = Link(ET.tz(0.4375)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.210)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.tz(0.2025)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.2755)*ET.Rz(), name="link5", parent=l4)
        # l5 = Link(ET.tz(0.2755)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(ER6, self).__init__(elinks, name="ER6", manufacturer="Rokae")  
        
class TestER6:
    def __init__(self,robot):
        self.robot = robot
        self.l12 = 0.404
        self.l23 = 0.4375
        self.l35 = 0.4125
        self.l56 = 0.2755
        self.tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
    def get_rotation(self,pose):
        return pose.R
    
    def get_position(self,pose):
        return pose.A[:3,3]
        
    def is_equal(self,l1,l2):
        ret = True
        if len(l1) == len(l2):
            for i in range(len(l1)):
                if abs(l1[i]-l2[i]*180/pi)>1e-06:
                    ret = False
        return ret

    def ink_a(self,joints,x,y,z,joint_index):
        j1 = joint_index[0]
        j2 = joint_index[1]
        j3 = joint_index[2]
        j4 = joints[3]
        j5 = joints[4]
        j6 = joints[5]
        T34 = np.mat([[cos(j4),-sin(j4),0,0],[sin(j4),cos(j4),0,0],[0,0,1,self.l35],[0,0,0,1]])
        T45 = np.mat([[cos(j5),0,sin(j5),0],[0,1,0,0],[-sin(j5),0,cos(j5),0],[0,0,0,1]])
        T56 = np.mat([[cos(j6),-sin(j6),0,0],[sin(j6),cos(j6),0,0],[0,0,1,self.l56],[0,0,0,1]])
        T35 = np.dot(T34,T45)
        T36 = np.dot(T35,T56)
        T3t = np.dot(T36,self.tool_frame)
        Px,Py,Pz = T3t[0,3],T3t[1,3],T3t[2,3]
        # print(f"T3t的位置={Px,Py,Pz}")
        # 计算joint1
        if abs(Py/np.sqrt(x**2+y**2)) > 1:
            print(f"轴1出现不可解的数学问题,{Py,np.sqrt(x**2+y**2)}")
            return False
        else:
            a1 = np.arcsin(Py/np.sqrt(x**2+y**2))
            b1 = np.arctan2(y,-x)
            # print(f"a1,b1={a1,b1}")
            # print(f"pi-a1={np.pi-a1}")
            theta1 = self.second_sort(a1,b1,j1)            
            # print(f"theta1={theta1}")
        # 计算joint2
        m1 = (x+sin(theta1)*Py)/cos(theta1)
        m2 = z - self.l12
        n = (Px**2+Pz**2-self.l23**2-m1**2-m2**2)/(-2*self.l23)
        # print(f"n的上半部分={Px**2+Py**2-self.a2**2-m1**2-m2**2}")
        #print(f"m1={m1},m2={m2},n={n}")
        if abs(n/np.sqrt(m1**2+m2**2)) > 1:
            print(f"轴2出现不可解的数学问题")
            return False
        else:
            a2 = np.arcsin(n/np.sqrt(m1**2+m2**2))
            b2 = np.arctan2(m2,m1)
            #print(f"a2,b2={a2,b2}")
            theta2 = self.second_sort(a2,b2,j2)
        # print(f"theta2={theta2*R2D}")
        # 计算joint2-3
        k = m2 - cos(theta2)*self.l23
        if abs(k/np.sqrt(Px**2+Pz**2)) > 1:
            print(f"轴3出现不可解的数学问题")
            return False
        else:
            a3 = np.arcsin(k/np.sqrt(Px**2+Pz**2))
            b3 = np.arctan2(Pz,-Px)
            #print(f"k={k},a3={a3},b3={b3}")
            theta23 =  self.second_sort(a3,b3,j2+j3)
            # print(f"theta2={theta2*R2D},theta23={theta23*R2D}")
            theta3 = theta23-theta2
        # print(f"thetas={theta1,theta2,theta3}")
        # 输出
        theta1 = np.array(theta1)
        theta2 = np.array(theta2)
        theta3 = np.array(theta3)
        joint = [theta1,theta2,theta3,j4,j5,j6]
        # print(f"joint_ans={joint}")
        if self.judge_position(x,y,z,joint,joints) is True:
            print(f"解析正确!正确的joint={joint}")
            return joint
        else:
            # print(f"错误的joint={jnt}")
            eps23 = self.get_another(a3,b3,j2-j3)
            eps3 = theta2-eps23
            joint = [theta1,theta2,eps3,j4,j5,j6]
            # print(f"eps23={eps23*R2D},eps3={eps3*R2D}")
            if self.judge_position(x,y,z,joint,joints) is True:
                print(f"解析正确!正确的joint={joint}")
                return joint
            else:
                print(f"反解位置错误!")
                # print(f"输出的joint={joint}")
                return False
     
    def replan_movetype(self,start_jnt,end_jnt,t):
        T0 = SE3(np.dot(self.set_jnt_angle(start_jnt),self.tool_frame))
        T1 = SE3(np.dot(self.set_jnt_angle(end_jnt),self.tool_frame))
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint_j = TJ.s.tolist() # 得到456轴
        TL = tr.ctraj(T0,T1,t) # tcp MoveL的直线
        joint_list = []
        for i in range(0,len(TL)):
            joint = joint_j[i]
            x = self.get_position(TL[i])[0]
            y = self.get_position(TL[i])[1]
            z = self.get_position(TL[i])[2]
            # print(f"第{i+1}条路径MoveL提供的位置{x,y,z}")
            if i == 0:
                joint_index = start_jnt
                joint_list.append(start_jnt)
            else:
                joint_index = joint_list[i-1]
                j = self.ink_a(joint,x,y,z,joint_index) 
                fk = robot.fkine(j)
                pk = self.get_position(SE3(np.dot(fk,self.tool_frame))) # 应该是tcp的位置
                if abs(pk[0]-x) > 1e-06 or abs(pk[1]-y) > 1e-06 or abs(pk[2]-z) > 1e-06:
                    print(f"位置错误{i}!")
                # js = np.array(j) 
                # print(f"解出的第{i}个点的角度={js*R2D}")
                joint_list.append(j)
                
        joint_list = joint_list[:-1]
        joint_list.append(end_jnt)
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
        
    def judge_position(self,x,y,z,joint,joints):
        T0 = robot.fkine(joint)
        posx = self.get_position(T0)
        print(f"flan={T0}")
        Tn = SE3(np.dot(robot.fkine(joint),self.tool_frame))
        posn = self.get_position(Tn)
        x1,y1,z1 = posn[0],posn[1],posn[2]
        print(f"原位置={x,y,z}")
        print(f"验算位置={Tn}")
        if abs(x1-x)>1e-06 or abs(y1-y)>1e-06 or abs(z1-z)>1e-06:
            print(f"原位置={x,y,z}")
            print(f"验算位置={x1,y1,z1}")
            js=joint
            for i in range(len(joint)):
                js[i] = js[i]*R2D
                joints[i] = joints[i]*R2D
            # print(f"输入角度:{joints}")
            print(f"新解角度:{js}")
            return False
        else:
            return True
    
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
   
    def space_line(self,p1,p2,t):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dz = p2[2]-p1[2]
        d = np.array([dx,dy,dz])
        point = p1+0.01*t*d
        return point
 
    def second_sort(self,alpha,beta,joint):
        # 已知theta = alpha-beta,beta唯一，alpha可能等于pi-alpha
        #print(f"alpha,beta,joint_index={alpha,beta,joint}") # 
        gamma = np.pi-alpha
        joint1 = alpha-beta
        joint2 = gamma-beta
        # print(f"joint1,joint2={joint1,joint2}")
        # 转化到[-180,180]之间
        joint1 = np.arctan2(np.sin(joint1),np.cos(joint1))
        joint2 = np.arctan2(np.sin(joint2),np.cos(joint2))
        diff1 = abs(joint-joint1)
        diff2 = abs(joint-joint2)
        # print(f"joint1,joint2={joint1,joint2}")
        # print(f"diff={diff1},{diff2}")
            
        if diff1<diff2:
            return joint1
        else:
            return joint2
        
    def judge_position(self,x,y,z,joint,joints):
        T0 = robot.fkine(joint)
        posx = self.get_position(T0)
        print(f"flan={T0}")
        Tn = SE3(np.dot(robot.fkine(joint),self.tool_frame))
        posn = self.get_position(Tn)
        x1,y1,z1 = posn[0],posn[1],posn[2]
        print(f"原位置={x,y,z}")
        print(f"验算位置={Tn}")
        if abs(x1-x)>1e-06 or abs(y1-y)>1e-06 or abs(z1-z)>1e-06:
            print(f"原位置={x,y,z}")
            print(f"验算位置={x1,y1,z1}")
            js=joint
            for i in range(len(joint)):
                js[i] = js[i]*R2D
                joints[i] = joints[i]*R2D
            # print(f"输入角度:{joints}")
            print(f"新解角度:{js}")
            return False
        else:
            return True
    

def test_link(robot):
    Robot = TestER6(robot)    
    # js = np.array([0,0,90,0,90,45])*D2R
    # js = np.array([73,-30,65,45,90,0])*D2R
    # js = np.array([82,-18,152,-99,72,83])*D2R
    js = np.array([-52,92,74,132,32,142])*D2R
    fk_start = robot.fkine(js)
    # fk_start = np.dot(fk_start,tool_frame)
    # print(f"fk_start={fk_start}")
    pos = Robot.get_position(fk_start)
    x,y,z = pos[0],pos[1],pos[2]
    j1 = js[0]
    j2 = js[1]
    j3 = js[2]
    j4 = js[3]
    j5 = js[4]
    j6 = js[5]
    l12 = 0.404
    l23 = 0.4375
    l35 = 0.4125
    l56 = 0.2755
    T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,0],[0,0,0,1]])
    T12 = np.mat([[cos(j2),0,sin(j2),0],[0,1,0,0],[-sin(j2),0,cos(j2),l12],[0,0,0,1]])
    T23 = np.mat([[cos(j3),0,sin(j3),0],[0,1,0,0],[-sin(j3),0,cos(j3),l23],[0,0,0,1]])
    T34 = np.mat([[cos(j4),-sin(j4),0,0],[sin(j4),cos(j4),0,0],[0,0,1,l35],[0,0,0,1]])
    T45 = np.mat([[cos(j5),0,sin(j5),0],[0,1,0,0],[-sin(j5),0,cos(j5),0],[0,0,0,1]])
    T56 = np.mat([[cos(j6),-sin(j6),0,0],[sin(j6),cos(j6),0,0],[0,0,1,l56],[0,0,0,1]])
    # T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,l12],[0,0,0,1]])
    # T12 = np.mat([[sin(j2),cos(j2),0,0],[0,0,1,0],[cos(j2),-sin(j2),0,0],[0,0,0,1]])
    # T23 = np.mat([[sin(j3),cos(j3),0,l23],[-cos(j3),sin(j3),0,0],[0,0,1,0],[0,0,0,1]])
    # T34 = np.mat([[cos(j4),-sin(j4),0,0],[0,0,1,l35],[-sin(j4),-cos(j4),0,0],[0,0,0,1]])
    # T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,-1,0],[sin(j5),0,cos(j5),0],[0,0,0,1]])
    # T56 = np.mat([[-cos(j6),sin(j6),0,0],[0,0,1,l56],[sin(j6),cos(j6),0,0],[0,0,0,1]])
    T02 = np.dot(T01,T12)
    T03 = np.dot(T02,T23)
    T04 = np.dot(T03,T34)
    T05 = np.dot(T04,T45)
    T06 = np.dot(T05,T56)
    T0t = np.dot(T06,tool_frame)
    # print(f"T0t={T0t}")
    
    ja = Robot.ink_a(js,x,y,z,js)
    ja = np.array(ja)*R2D
    print(f"ja={ja}")

def test_move(robot):
    Robot = TestER6(robot)    
    js = np.array([0,0,90,0,90,45])*D2R
    je = np.array([-52,92,74,132,32,142])*D2R
    fk_start = robot.fkine(js)
    fk_end = robot.fkine(je)
    
    
if __name__ == "__main__":
    robot = ER6()
    test_link(robot)
    