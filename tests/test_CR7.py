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
#tool_frame = np.array([[1,0,0,0.00459],[0,1,0,-0.00182],[0,0,1,0.48267],[0,0,0,1]])
#tool_frame1 = np.array([[0.45782587,-0.02580994,-0.88866716,0.00459],[-0.00663264,-0.99964985,0.02561624,-0.00182],[-0.88901714,-0.00583357,-0.45783675,0.48267],[0,0,0,1]])

class CR7(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.296)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.490)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.360)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.ty(0.150)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.127)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(CR7, self).__init__(elinks, name="CR7", manufacturer="Rokae")  

class CR7t(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.296)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.490)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        elinks = [l0,l1,l2]
        super(CR7t, self).__init__(elinks, name="CR7t", manufacturer="Rokae")  

class CR7b(ERobot):
    def __init__(self):
        l3 = Link(ET.tz(0.360)*ET.Rz(), name="link3", parent=None)
        l4 = Link(ET.ty(0.150)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.127)*ET.Rz(), name="link5", parent=l4)
        elinks = [l3,l4,l5]
        super(CR7b, self).__init__(elinks, name="CR7b", manufacturer="Rokae")  

class TestCR7:
    def __init__(self,robot):
        self.robot = robot
        self.d1 = 0.296
        self.a2 = 0.490
        self.d4 = 0.360
        self.d5 = 0.150
        self.d6 = 0.127
        # self.tool_frame = np.array([[1,0,0,0.00459],[0,1,0,-0.00182],[0,0,1,0.48267],[0,0,0,1]])
        #self.tool_frame = np.array([[0.45782587,-0.02580994,-0.88866716,0.00459],
                                    # [-0.00663264,-0.99964985,0.02561624,-0.00182],
                                    # [-0.88901714,-0.00583357,-0.45783675,0.48267],
                                    # [0,0,0,1]])
        self.tool_frame = np.array([[1,0,0,0.088],[0,1,0,-0.00233],[0,0,1,0.04188],[0,0,0,1]])

    def quat2eular(self,quat):
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)
        return euler
        
    def rotm2eular(self,RT) :
        r = R.from_matrix(RT)
        return r.as_euler('xyz', degrees=False)
            
    def eular2rotm(self,eular):
        R_x = np.array([[1, 0, 0],
                    [0, math.cos(eular[0]), -math.sin(eular[0])],
                    [0, math.sin(eular[0]), math.cos(eular[0])]
                    ])
 
        R_y = np.array([[math.cos(eular[1]), 0, math.sin(eular[1])],
                        [0, 1, 0],
                        [-math.sin(eular[1]), 0, math.cos(eular[1])]
                        ])
    
        R_z = np.array([[math.cos(eular[2]), -math.sin(eular[2]), 0],
                        [math.sin(eular[2]), math.cos(eular[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def rotm2quat(self,RT):
        r = R.from_matrix(RT)
        return r.as_quat()
     
    def eular2quat(self,eular):
        r = R.from_euler('xyz', eular, degrees=False)
        return r.as_quat()
         
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
        
    def ink_a(self,joints,x,y,z,joint_index):
        j1 = joint_index[0]
        j2 = joint_index[1]
        j3 = joint_index[2]
        j4 = joints[3]
        j5 = joints[4]
        j6 = joints[5]
        # print(f"j4-6={j4,j5,j6}")
        # 有问题，日后再修
        T34 = np.mat([[cos(j4),-sin(j4),0,0],[0,0,-1,-self.d4],[sin(j4),cos(j4),0,0],[0,0,0,1]])
        T45 = np.mat([[cos(j5),-sin(j5),0,0],[0,0,1,self.d5],[-sin(j5),-cos(j5),0,0],[0,0,0,1]])
        # T56 = np.mat([[cos(j6),-sin(j6),0,0],[0,0,-1,-self.d6],[sin(j6),cos(j6),0,0],[0,0,0,1]])
        T56 = np.mat([[-cos(j6),sin(j6),0,0],[0,0,-1,-self.d6],[-sin(j6),-cos(j6),0,0],[0,0,0,1]])
        T35 = np.dot(T34,T45)
        T36 = np.dot(T35,T56)
        T3t = np.dot(T36,self.tool_frame)
        Px,Py,Pz = T3t[0,3],T3t[1,3],T3t[2,3]
        print(f"T3t的位置={Px,Py,Pz}")
        # 计算joint1
        if abs(Pz/np.sqrt(x**2+y**2)) > 1:
            print(f"轴1出现不可解的数学问题,{Pz,np.sqrt(x**2+y**2)}")
            return False
        else:
            a1 = np.arcsin(Pz/np.sqrt(x**2+y**2))
            b1 = np.arctan2(-y,x)
            #print(f"a1,b1={a1,b1}")
            # print(f"pi-a1={np.pi-a1}")
            theta1 = self.second_sort(a1,b1,j1)
            # print(f"theta1={theta1}")
        # 计算joint2
        m1 = (x-sin(theta1)*Pz)/cos(theta1)
        m2 = z - self.d1
        n = (Px**2+Py**2-self.a2**2-m1**2-m2**2)/(-2*self.a2)
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
        k = m2 - cos(theta2)*self.a2
        if abs(k/np.sqrt(Px**2+Py**2)) > 1:
            print(f"轴3出现不可解的数学问题")
            return False
        else:
            a3 = np.arcsin(k/np.sqrt(Px**2+Py**2))
            b3 = np.arctan2(-Py,Px)
            #print(f"k={k},a3={a3},b3={b3}")
            theta23 =  self.second_sort(a3,b3,j2-j3)
            # print(f"theta2={theta2*R2D},theta23={theta23*R2D}")
            theta3 = theta2-theta23
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
            q = self.rotm2quat(R1)
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
            q = self.rotm2quat(R1)
            vector,angle = self.quat2axis(q)
            ang.append(angle)
            vec.append(vector)
        return ang
    
    def quat2axis(self,quat):
        x = quat[1]
        y = quat[2]
        z = quat[3]
        w = quat[0]
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
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotm2eular(R1)
            E1.append(eular1)
            # print(f"E1={eular1}")
        E1.pop(0)
        E1.insert(0,E1[0])
        return E1
    
    def compare_rpyl(self,start_jnt,end_jnt,t):
        joint_list1 = self.moveL(start_jnt,end_jnt,t)
        # joint_list2 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotm2eular(R1)
            E1.append(eular1)
            # print(f"E1={eular1}")
        E1.pop(0)
        E1.insert(0,E1[0])
        return E1
    
    def compare_rpyj(self,start_jnt,end_jnt,t):
        joint_list1 = self.moveAbsJ(start_jnt,end_jnt,t)
        E1 = []
        # 提取rotation matrix，然后分别计算rpy，对比，相减
        for i in range(0,len(t)):
            jnt1 = joint_list1[i]
            fk1 = robot.fkine(jnt1)
            R1 = self.get_rotation(fk1)
            eular1 = self.rotm2eular(R1)
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
            q = self.rotm2quat(R1)
            vector,angle = self.quat2axis(q)
            ang.append(angle)
            vec.append(vector)
        return ang
    
    def moveL(self,start_jnt,end_jnt,t):
        Ts = SE3(np.dot(self.set_jnt_angle(start_jnt),self.tool_frame))
        Te = SE3(np.dot(self.set_jnt_angle(end_jnt),self.tool_frame))
        # print(f"起始点={ps}")
        TL = tr.ctraj(Ts,Te,t)
        joint=[]
        point=[]
        for i in range(0,len(TL)):
            p = self.get_position(TL[i])
            if i == 0:
                j = self.get_joint(TL[i],start_jnt)
            else:
                j = self.get_joint(TL[i],joint[i-1])
            joint.append(j)
            point.append(p)
            # print(f"第{i}个MoveL路径={j}")
        # print(f"终止点={pe}")
        return joint,point
    
    def moveAbsJ(self,start_jnt,end_jnt,t):
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint = TJ.s.tolist()
        # robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
        return joint
       
    def compare_aa(self,start_jnt,end_jnt,t):
        diff = []
        a1 = self.compare_axis_angle(start_jnt,end_jnt,t)
        a2 = self.compare_movel_angle(start_jnt,end_jnt,t)
        for i in range(len(a1)):
            dif = abs(a1[i]-a2[i])
            diff.append(dif)
        return diff
  
    def tcp_to_flan_base(self,tcp,eular,tool_frame):
        # T0t = T06*tool_frame
        # eular+pose->T0t
        R = self.eular2rotm(eular)
        bot = np.array([[0,0,0,1]])
        T = np.array([[tcp[0]],[tcp[1]],[tcp[2]]])
        e = np.concatenate((R,T),axis=1)
        T0t = np.concatenate((e,bot),axis=0)
        # T0t = T06*tool_frame
        # print(f"T0t={T0t}")
        T06 = np.dot(T0t,np.linalg.inv(tool_frame))
        return T06
    
    def replan2_move(self,jnt_list,pnt_list):
        output_joint = []
        if len(jnt_list) != len(pnt_list):
            print(f"数组长度错误,角度数组长={len(jnt_list)},位置数组长={len(pnt_list)}")
            return False
        for i in range(len(jnt_list)):
            # 这里的xyz都是tcp位置
            x = pnt_list[i][0]
            y = pnt_list[i][1]
            z = pnt_list[i][2]
            if i == 0:
                    joint_index = jnt_list[0]
            else:
                joint_index = jnt_list[i-1]
            j = self.ink_a(jnt_list[i],x,y,z,joint_index) 
            fk = robot.fkine(j)
            pk = self.get_position(SE3(np.dot(fk,self.tool_frame)))
            if abs(pk[0]-x) > 1e-04 or abs(pk[1]-y) > 1e-04 or abs(pk[2]-z) > 1e-04:
                print(f"位置不对2")
                return False
            output_joint.append(j)
        return output_joint
    
    def space_line(self,p1,p2,t):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dz = p2[2]-p1[2]
        d = np.array([dx,dy,dz])
        point = p1+0.01*t*d
        return point
        
    def trans_in2_pi(self,jnt):
        rotvec = R.as_rotvec()
        return rotvec
    
def test_CR7(robot):
    Robot = TestCR7(robot)
    # start_jnt = np.array([0,0,90,0,90,0])*D2R
    # start_jnt = np.array([0,0.365063,-1.46438,0,1.5708,0])
    # start_jnt = np.array([82,-18,152,-99,72,164])*D2R
    # start_jnt = np.array([1.64303,-0.023845,2.30599,-2.46314,0.539396,1.96355])
    start_jnt = np.array([1.90615,-0.0853887,1.76006,-0.743045,0.570879,-0.541489])
    ans = start_jnt
    print(f"输入的测试角度,取其456轴={ans}")
    fk = robot.fkine(start_jnt)
    pos = Robot.get_position(SE3(np.dot(fk,Robot.tool_frame)))
    print(f"T0t的位置={pos}")
    joint = Robot.ink_a(start_jnt,pos[0],pos[1],pos[2],start_jnt)
    print(f"逆解输出的角度={np.array(joint)}")
    
def testLink():
    robo = CR7()
    robo1 = CR7t()
    robo2 = CR7b()
    # jnt = np.array([0,0,90,0,90,0])*D2R
    jnt = np.array([82,-18,152,-99,72,164])*D2R
    jntt = jnt[0:2]
    jntb = jnt[3::]
    fk = robo.fkine(jnt)
    j1,j2,j3 = jnt[0],jnt[1],jnt[2]
    j4,j5,j6 = jnt[3],jnt[4],jnt[5]

    fkt = robo1.fkine(jntt)
    fkb = robo2.fkine(jntb)
    d1,a2 = 0.296,0.490
    d4,d5,d6 = 0.360,0.150,0.127
    
    T03 = np.mat([[-cos(j1)*cos(j2-j3),-cos(j1)*sin(j2-j3),sin(j1),cos(j1)*sin(j2)*a2],
                  [-sin(j1)*cos(j2-j3),-sin(j1)*sin(j2-j3),-cos(j1),sin(j1)*sin(j2)*a2],
                  [sin(j2-j3),-cos(j2-j3),0,d1+cos(j2)*a2],
                  [0,0,0,1]])
    
    T36 = np.mat([[cos(j4)*cos(j5)*cos(j6)-sin(j4)*sin(j6), -cos(j4)*cos(j5)*sin(j6)-sin(j4)*cos(j6), cos(j4)*sin(j5), cos(j4)*sin(j5)*d6-sin(j4)*d5],
                  [sin(j4)*cos(j5)*cos(j6)+cos(j4)*sin(j6), -sin(j4)*cos(j5)*sin(j6)+cos(j4)*cos(j6), sin(j4)*sin(j5), sin(j4)*sin(j5)*d6+cos(j4)*d5],
                  [-sin(j5)*cos(j6), sin(j5)*sin(j6), cos(j5), cos(j5)*d6+d4],
                  [0,0,0,1]])

    T06 = np.dot(fkt,fkb)
    T06dh = np.dot(T03,T36)
    # T0t = np.dot(T06,tool_frame)
    # T3t = np.dot(T36,tool_frame)
    # T3b = np.dot(fkb,tool_frame)

    print(f"整体连杆={fk}")
    # print(f"DH参数连杆重构={T06}")
    # print(f"tcp位置={T0t}")
    
    # print(f"T03的连杆参数={fkt}")
    # print(f"T03的DH参数={T03}")
    
    print(f"T06的连杆参数={T06}")
    # print(f"T06的DH参数={T06dh}")

    # print(f"T36的DH参数的mat={T36}")
    # print(f"T3t的位置={T3t}")
    
    # print(f"T36的连杆参数={fkb}")
    # print(f"T3t的位置={T3b}")
    
def test_melt(robot):
    Robot = TestCR7(robot)
    j1 = np.array([-55.449,38.440,-116.095,-31.074,-10.298,62.921])*D2R
    j2 = np.array([-51.767,39.642,-115.014,-33.196,-10.162,35.052])*D2R
    j3 = np.array([-54.742,39.182,-104.915,2.924,-17.555,-79.655])*D2R
    j4 = np.array([-56.481,37.876,-108.630,-1.585,-16.560,-102.362])*D2R
    t = np.linspace(0,1,100)
    # 
    path1 = Robot.replan_movetype(j1,j2,t)
    path2 = Robot.replan_movetype(j2,j3,t)
    path3 = Robot.replan_movetype(j3,j4,t)
    paths = path1+path2+path3
    te = np.linspace(0,1,len(paths)) 
    # xplot(np.array(paths),block=True)
    # robot.plot(np.array(paths),backend='pyplot',dt=0.01,block=True)
    
    E1 = []
    ### 提取rotation matrix，然后分别计算rpy，对比，相减
    for i in range(0,len(paths)):
        jnt1 = paths[i]
        fk1 = SE3(robot.fkine(jnt1)*tool_frame)
        R1 = Robot.get_rotation(fk1)
        eular1 = Robot.rotm2eular(R1)
        E1.append(eular1)
    E1.pop(0)
    E1.insert(0,E1[0])

    for i in range(len(E1)):
        E1[i] = E1[i]*R2D
        
    # plt.title("CR7 Wrist Singular Fix Path RPY")
    # plt.plot(te,E1)
    # plt.ylabel("deg")
    # plt.legend(['r','p','y'], loc="upper left")
    # plt.show()
    
    ang = []
    quads = []
    vecs = []
    for i in range(0,len(paths)):
        jnt1 = paths[i]
        fk1 = SE3(np.dot(robot.fkine(jnt1),tool_frame)) 
        R1 = Robot.get_rotation(fk1)
        q = Robot.rotm2quat(R1)
        qs = q
        quads.append(qs)
        vector,angle = Robot.quat2axis(qs)
        ang.append(angle)
        vecs.append(vector)
    
    q1 = Robot.rotm2quat(Robot.get_rotation(SE3(np.dot(robot.fkine(j1),tool_frame))))
    q2 = Robot.rotm2quat(Robot.get_rotation(SE3(np.dot(robot.fkine(j2),tool_frame))))
    q3 = Robot.rotm2quat(Robot.get_rotation(SE3(np.dot(robot.fkine(j3),tool_frame))))
    q4 = Robot.rotm2quat(Robot.get_rotation(SE3(np.dot(robot.fkine(j4),tool_frame))))
    q = [q1,q2,q3,q4]
    quats = []
    euler = []
    for i in range(len(q)-1):
        key_times = [0,100]
        times = np.linspace(0,99,100)
        # angle = np.dot(q[i],q[i+1])
        # print(f"angle = {angle}")
        rots = R.from_quat([q[i],q[i+1]])
        slerp = Slerp(key_times, rots)
        interp_rots = slerp(times)
        # print(f"interp_rots{i}={interp_rots.as_quat()}")
        for j in range(len(interp_rots)):
            # quats.append(np.array(interp_rots[i].as_quat))
            a = interp_rots[j].as_quat()
            quats.append(a)
            b = interp_rots[j].as_euler('xyz', degrees=True)
            euler.append(b)
            
    difex = [] 
    difey = []
    difez = []
    for i in range(len(euler)):
        # print(f"第{i}个欧拉角={euler[i]}")
        dif1 = abs(euler[i][0]-E1[i][0])
        dif2 = abs(euler[i][1]-E1[i][1])
        dif3 = abs(euler[i][2]-E1[i][2])
        difex.append(dif1)
        difey.append(dif2)
        difez.append(dif3)
        
    tq = np.linspace(0,1,len(euler))
    # plt.title("Path Euler Comparision")
    # plt.plot(te,euler,'b',label="Slerp")
    # plt.plot(te,E1,'r',label="Wrist")
    # plt.ylabel("deg")
    # plt.show()
    
    # plt.title("Path Euler Difference")
    # plt.plot(te,difex,'r',label="euler_x")
    # plt.plot(te,difey,'y',label="euler_y")
    # plt.plot(te,difez,'b',label="euler_z")
    # plt.legend()
    # plt.ylabel("deg")
    # plt.show()
        
    # plt.title("Quads Path")
    # plt.plot(te,quats,'b')
    # plt.plot(te,quads,'r')
    # plt.show()
    ane = []
    vec = []
    for i in range(len(quats)):
        v,ans = Robot.quat2axis(quats[i])
        ane.append(ans)
        vec.append(v)
        
    # plt.title("CR7 Melting Short Path Provided by Wanhao's Axis-Angle")
    # plt.plot(te,ang,'r')
    # plt.plot(te,ane,'b')
    # plt.legend(['Wrist','Slerp'])
    # plt.ylabel("deg")
    # plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xa,ya,za = [],[],[]
    xb,yb,zb = [],[],[]
    for i in range(len(vecs)):
        vs = np.array([vecs[i][0],vecs[i][1],vecs[i][2]])
        norm_x = np.linalg.norm(vs, axis=0)
        # print(ang[i]*D2R)
        # q = ax.quiver(0,0,0, vecs[i][0]*ang[i]*D2R, vecs[i][1]*ang[i]*D2R,vecs[i][2]*ang[i]*D2R,color='b') 
        # q = ax.quiver(0,0,0, vecs[i][0]*ang[i]*D2R, vecs[i][1]*ang[i]*D2R,vecs[i][2]*ang[i]*D2R,length=1,arrow_length_ratio=0.01, normalize=True)
        xa.append(vecs[i][0]*ang[i]*D2R)
        ya.append(vecs[i][1]*ang[i]*D2R)
        za.append(vecs[i][2]*ang[i]*D2R)
        xb.append(vec[i][0]*ane[i]*D2R)
        yb.append(vec[i][1]*ane[i]*D2R)
        zb.append(vec[i][2]*ane[i]*D2R)
        # 蓝色为Slerp，红色为Wrist
    # ax.scatter(vecs[i][0]*ang[i]*D2R, vecs[i][1]*ang[i]*D2R,vecs[i][2]*ang[i]*D2R,color = 'r')
    # ax.scatter(vec[i][0]*ane[i]*D2R, vec[i][1]*ane[i]*D2R,vec[i][2]*ane[i]*D2R,color = 'b')
    ax.scatter(xa,ya,za,c='r',label='Wrist')
    ax.scatter(xb,yb,zb,c='b',label='Slerp')
    ax.set_xlim3d(-1.5, 1.5)
    ax.set_ylim3d(0, 0.5)
    ax.set_zlim3d(0, 0.5)
    plt.show()
    
    # dif = []
    # plt.title("CR7 Melting Short Path Provided by Wanhao's Axis-Angle Difference")
    # for i in range(len(ane)):
    #     d = abs(ane[i]-ang[i])
    #     if i == 120:
    #         print(f"第{i}个轴角,Slerp={ane[i]},Wrist={ang[i]}")
    #         print(f"第{i}个轴角的差值={d}")
    #         print(f"第{i}个欧拉角,Slerp={euler[i]},Wrist={E1[i]}")
    #         print(f"第{i}个欧拉角差距={difex[i],difey[i],difez[i]}")    
    #         print(f"第{i}个四元数,Slerp={quats[i]},Wrist={quads[i]}")
    #     dif.append(d)
    # plt.plot(te,dif,'r')
    # plt.ylabel("deg")
    # plt.show()
    
def get_tool_frame(robot):
    Robot = TestCR7(robot)
    mat = Robot.eular2rotm([-162.77*D2R,49.94*D2R,-23.10*D2R])
    # [x,y,z]=[0.00459,-0.00182,0.48267]
    print(mat)
    # j1 = np.array([-55.449,38.440,-116.095,-31.074,-10.298,62.921])*D2R
    # fk1 = SE3(robot.fkine(j1))
    # # fk2 = SE3(robot.fkine(j1)*tool_frame2)
    # print(type(fk1))
    # # print(type(fk2))
    
def test_trans(robot):
    Robot = TestCR7(robot)
    j1 = np.array([-55.449,38.440,-116.095,-31.074,-10.298,62.921])*D2R
    fk1 = robot.fkine(j1)*tool_frame
    rotm = Robot.get_rotation(SE3(fk1))
    e1 = Robot.rotm2eular(rotm)
    q1 = Robot.rotm2quat(rotm)
    # print(f"旋转矩阵={rotm},欧拉角={e1},四元数={q1}")
    # rpy转rotm
    rotn = Robot.eular2rotm(e1)
    # quad转rpy
    e2 = Robot.quat2eular(q1)
    # rpy转quad
    q2 = Robot.eular2quat(e1)
    # print(f"逆旋转矩阵={rotn},四元数和欧拉角互转={e2},{q2}")
    # 结论，四元数，旋转矩阵，欧拉角之间的转换无误
    
    ea = np.array([171.33562095,-16.18207361,-38.22084353])*D2R
    eb = np.array([173.8467719,-16.50245171,-49.08644919])*D2R
    qa = Robot.eular2quat(ea)
    qb = Robot.eular2quat(eb)
    print(f"qa={qa},qb={qb}")
    aa = Robot.quat2axis(qa)
    ab = Robot.quat2axis(qb)
    print(f"aa={aa},ab={ab}")  
    #Slerp=[ 0.98651569 -0.00754865  0.08881301  0.1372664 ],Wrist=[0.98329131 0.09400221 0.07469213 0.13683152]  
    
def test_quat(robot):
    Robot = TestCR7(robot)
    robo1 = CR7t()
    j1 = np.array([-0.895666,0.694665,-1.98269,-0.521523,-0.199213,0.560254])
    fk1 = robot.fkine(j1)*tool_frame1
    # print(f"fk1={fk1}")
    x,y,z = 0.29477,-0.674952,-0.198377
    jnt = Robot.ink_a(j1,x,y,z,j1)
    robot.plot(np.array(jnt),backend='pyplot',dt=0.01,block=True)
    j = np.array(jnt)*R2D
    print(f"jnt={j}")
    # fk = robot.fkine(jnt)
    # print(f"fk={fk}")
    # ft = fk*tool_frame1
    # print(f"ft={ft}")
    # joint=np.array([-0.90011274, 0.69168809,-1.99387473, -0.521523, -0.199213, 0.560254])
    # kk = robot.fkine(joint)
    # print(f"kk={kk}")
    # j2 = np.array([0,0,90,0,90,0])*D2R
    # jntt = j2[0:2]
    # jntb = j2[3::]
    # fkt = robo1.fkine(jntt)
    # print(f"fkt={fkt}")
    # fk2 = robot.fkine(j2)
    # print(f"fk2={fk2}")
  
def test_road(robot):
    Robot = TestCR7(robot)
    t = np.linspace(0,1,100)
    st = np.array([-39.34,-51.06,149.03,-32.88,-63.23,38.08])*D2R
    et = np.array([-49.07,-27.24,-142.89,-31.68,-39.65,39.30])*D2R
    joint = Robot.replan_movetype(st,et,t)
    robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
    for i in range(len(joint)):
        print(f"joint{i}={np.array(joint[i])*R2D}")
    xplot(np.array(joint),block=True)
       
if __name__ == "__main__":
    robot = CR7()
    # test_CR7(robot)
    testLink()
    # test_melt(robot)
    # get_tool_frame(robot)
    # test_trans(robot)
    # test_quat(robot)
    # test_road(robot)