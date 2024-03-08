import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy import linalg
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
from scipy.interpolate import interpn as interpn
from scipy.spatial.transform import Slerp
from spatialmath.base import tr2x, numjac, numhess
from scipy.linalg import block_diag

pi = np.pi
R2D = 180/pi
D2R = pi/180
tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])


class SR4t(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.328)*ET.Ry(), name="link1", parent=l0)
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
        
class SR4(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.328)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.400)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.400)*ET.tx(0.05)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.ty(-0.136)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.1035)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(SR4, self).__init__(elinks, name="SR4", manufacturer="Rokae")  

class TestSR4:
    def __init__(self,robot):
        self.robot = robot
        # self.d1 = 0.355
        self.d1 = 0.328
        self.dx = 0.4
        self.dy = -0.05
        self.tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
        
    def quat2eular(self,quat):
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=True)
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
     
    def set_jnt_angle(self,start_angle):
        pose = self.robot.fkine(start_angle)
        return pose

    def get_rotation(self,pose):
        return pose.R
    
    def get_position(self,pose):
        return pose.A[:3,3]
    
    def create_matrix(self, R, T):
        T = np.eye(4,4)
        for i in range(3):
            for j in range(3):
                T[i][j] = R[i][j]
            T[i][3] = T[i]
        return T
        
    def get_joint(self,RT,jnt):
        # joint = robot.ikine_LM(RT,q0=jnt).q
        # joint = robot.ikine_min(RT,jnt).q
        joint = robot.ikine_LMS(RT,jnt).q
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
        # print(f"输入解析解的absj角度={j4*R2D,j5*R2D,j6*R2D}")
        cos = np.cos
        sin = np.sin
        # 通过连杆计算T36
        rob_back = SR4b()
        T36 = rob_back.fkine(joints[3:5])
        T3t = SE3(np.dot(T36,self.tool_frame))
        pose = self.get_position(T3t)
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
                print(f"解析正确!")
                return joint
            else:
                print(f"反解位置错误!")
                print(f"输出的joint={joint}")
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
        Tn = SE3(np.dot(robot.fkine(joint),self.tool_frame))
        posn = self.get_position(Tn)
        x1,y1,z1 = posn[0],posn[1],posn[2]
        if abs(x1-x)>1e-06 or abs(y1-y)>1e-06 or abs(z1-z)>1e-06:
            print(f"原位置={x,y,z}")
            print(f"验算位置={x1,y1,z1}")
            js=joint
            for i in range(len(joint)):
                js[i] = js[i]*R2D
                joints[i] = joints[i]*R2D
            print(f"输入角度:{joints}")
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
        point = p1+0.002*t*d
        return point
        
    def trans_in2_pi(self,jnt):
        # 需要rad
        joint = [0,0,0,0,0,0]
        for i in range(len(jnt)):
            joint[i] = np.arctan2(np.sin(jnt[i]),np.cos(jnt[i]))
        return joint

def test_fix(robot):
    t = np.linspace(0,1,100) 
    Robot = TestSR4(robot)
    j1 = np.array([-1.66784,-0.244627,-2.38704,0.619301,-0.264884,1.22412])
    j2 = np.array([-1.63023,-0.272453,-2.47516,0.77765,-0.220629,1.0262])
    j3 = np.array([-1.7009,-0.256954,-2.45228,-0.626027,0.345847,2.18662])
    j4 = np.array([-1.77799,-0.199206,-2.61334,-0.399962,0.653154,1.84284])
    # p1 = Robot.get_position(robot.fkine(j2))
    ps = Robot.get_position(SE3(np.dot(robot.fkine(j2),tool_frame)))
    p2 = np.array([0.105475,-0.511938,0.422807])
    p3 = np.array([0.102627,-0.501553,0.406773])
    pe = Robot.get_position(SE3(np.dot(robot.fkine(j3),tool_frame)))
    
    e1 = Robot.rotm2eular(Robot.get_rotation(robot.fkine(j1)))
    e2 = Robot.rotm2eular(Robot.get_rotation(robot.fkine(j2)))
    e3 = Robot.rotm2eular(Robot.get_rotation(robot.fkine(j3)))
    # print(f"eulars = {e1,e2,e3}")
    
    # path1,ppt = Robot.moveL(j1,j2,t)
    # path5,ppt = Robot.moveL(j3,j4,t)
    path1 = Robot.replan_movetype(j1,j2,t)
    path5 = Robot.replan_movetype(j3,j4,t)
    t2 = np.linspace(0,1,300) 
    joint_j = Robot.moveAbsJ(j2,j3,t2)
    joint_a = joint_j[0:100]
    joint_b = joint_j[100:200]
    joint_c = joint_j[200::]
    point_a = []
    point_b = []
    point_c = []
    for i in range(len(t)):
        pos1 = Robot.space_line(ps,p2,i)
        pos2 = Robot.space_line(p2,p3,i)
        pos3 = Robot.space_line(p3,pe,i)
        point_a.append(pos1)
        point_b.append(pos2)
        point_c.append(pos3)
    path2 = Robot.replan2_move(joint_a,point_a)
    path3 = Robot.replan2_move(joint_b,point_b)
    path4 = Robot.replan2_move(joint_c,point_c)
    joints = path1+path2+path3+path4+path5
    theta = path2+path3+path4
    # xplot(np.array(joints),block=True)
    # robot.plot(np.array(joints),backend='pyplot',dt=0.01,block=True)
    # for i in range(len(joint_j)):
    #     print(f"轴空间解{i}={joint_j[i]}")
    #     print(f"腕关节解{i}={theta[i]}")
    # 反解位置
    # ax1 = plt.axes(projection='3d')
    # ax1.set_title("Cart Point In 3D Space")
    # for i in range(len(joints)):
    #     pos = Robot.get_position(SE3(np.dot(robot.fkine(joints[i]),tool_frame)))
    #     ax1.plot3D(pos[0],pos[1],pos[2],'r-*')
    # plt.show()
    
    ##  测试姿态
    E1 = []
    ### 提取rotation matrix，然后分别计算rpy，对比，相减
    for i in range(0,len(joints)):
        jnt1 = joints[i]
        fk1 = robot.fkine(jnt1)
        R1 = Robot.get_rotation(fk1)
        eular1 = Robot.rotm2eular(R1)
        E1.append(eular1)
    E1.pop(0)
    E1.insert(0,E1[0])

    for i in range(len(E1)):
        E1[i] = E1[i]*R2D

    te = np.linspace(0,1,len(joints)) 
    plt.title("SR4 Wrist Singular Fix Path RPY")
    plt.plot(te,E1)
    plt.ylabel("deg")
    plt.legend(['r','p','y'], loc="upper left")
    plt.show()
        
    # 测试轴角
    ang = []
    quads = []
    for i in range(0,len(joints)):
        jnt1 = joints[i]
        fk1 = SE3(np.dot(robot.fkine(jnt1),tool_frame)) 
        R1 = Robot.get_rotation(fk1)
        q = Robot.rotm2quat(R1)
        qs = q
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        qs[0] = b
        qs[1] = -a
        qs[2] = d
        qs[3] = -c
        quads.append(qs)
        vector,angle = Robot.quat2axis(qs)
        ang.append(angle)
    # print(f"修正路径的四元数={quads}")

    
    # 转换rpy to quat, slerp,这里四元数的转换很迷，算出来顺序有些问题，需要重新构造顺序
    qa = Robot.rotm2quat(Robot.get_rotation(robot.fkine(j1)))
    qb = Robot.rotm2quat(Robot.get_rotation(robot.fkine(j2)))
    q1 = R.from_quat([0.91265597,-0.13327802,0.10038591,0.37312024])
    q2 = R.from_quat([0.91713986,-0.09976903,0.10154112,0.3722768])
    qi = [0,0,0,0]
    qj = [0,0,0,0]
    qi[0] = qa[1]
    qi[1] = -qa[0]
    qi[2] = qa[3]
    qi[3] = -qa[2]
    qj[0] = qb[1]
    qj[1] = -qb[0]
    qj[2] = qb[3]
    qj[3] = -qb[2]
    # print(f"qa={qi},q1={q1.as_quat()}")
    # print(f"qb={qj},q2={q2.as_quat()}")
    q3 = Robot.rotm2quat(Robot.eular2rotm(np.array([2.27535,-0.233094,-0.0320564])))
    q4 = Robot.rotm2quat(Robot.eular2rotm(np.array([2.06406,-0.175967,0.0474866])))
    # q5 = Robot.rotm2quat(Robot.get_rotation(robot.fkine(j3)))
    q5 = R.from_quat([0.81462229,-0.00287296,0.04935389,0.57788101])
    q6 = R.from_quat([0.7812075, 0.0134957, 0.0128575, 0.6239931])
    
    q = [q1.as_quat(),q2.as_quat(),q3,q4,q5.as_quat(),q6.as_quat()]
    quats = []
    euler = []
    for i in range(len(q)-1):
        key_times = [0,100]
        times = np.linspace(0,99,100)
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
            # print(f"q{i}={a}")
        # print(f"r={r.as_euler('xyz', degrees=True)}")

    difex = [] 
    difey = []
    difez = []
    for i in range(len(euler)):
        # print(f"第{i}个欧拉角={euler[i]}")
        if E1[i][2]<0:
            E1[i][2] = -E1[i][2]
        E1[i][2] = -(180-E1[i][2])
        E1[i][0] = -E1[i][0]
        E1[i][1] = -E1[i][1]
        dif1 = abs(euler[i][0]-E1[i][0])
        dif2 = abs(euler[i][1]-E1[i][1])
        dif3 = abs(euler[i][2]-E1[i][2])
        difex.append(dif1)
        difey.append(dif2)
        difez.append(dif3)
        
    tq = np.linspace(0,1,len(euler))
    plt.title("Path Euler Comparision")
    plt.plot(te,euler,'b')
    plt.plot(te,E1,'r')
    plt.ylabel("deg")
    plt.show()
    
    plt.title("Path Euler Difference")
    plt.plot(te,difex,'r',label="euler_x")
    plt.plot(te,difey,'y',label="euler_y")
    plt.plot(te,difez,'b',label="euler_z")
    plt.legend()
    plt.ylabel("deg")
    plt.show()
    
    ane = []
    # plt.title("Quads Path")
    # plt.plot(te,quats,'b')
    # plt.plot(te,quads,'r')
    # plt.show()
    # print(f"SLERP的四元数={quats[-1]}")
    # print(f"插补的四元数={quads[-1]}")
    
    # for i in range(len(quats)):
    #     v,ans = Robot.quat2axis(quats[i])
    #     ane.append(ans)
    # # print(f"axis={ane}")
    # plt.title("SR4 Fix Path Axis-Angle")
    # plt.plot(te,ang,'r')
    # plt.plot(te,ane,'b')
    # plt.ylabel("deg")
    # plt.show()
    
    # dif = []
    # plt.title("SR4 Fix Path Axis-Angle Difference")
    # for i in range(len(ane)):
    #     d = abs(ane[i]-ang[i])
    #     dif.append(d)
    # plt.plot(te,dif,'r')
    # plt.ylabel("deg")
    # plt.show()

def test_jacob(robot):
    q = np.array([1.79395,0.0873863,1.84406,0.0533822,0.488315,-0.503091])
    TE = robot.fkine(q)
    J0 = numjac(lambda q: robot.fkine(q).A, q, SE=3)
    Je = block_diag(TE.R.T, TE.R.T) @ J0
    Tj = robot.jacobe(q)
    Ti = robot.jacob0(q)
    print(f"机器人工具箱jacobe={Tj}")
    # print(f"机器人工具箱jacob0={Ti}")
    # print(f"数据解算jacobe={Je}")
    jac_det = np.linalg.matrix_rank(Tj)#返回矩阵的秩
    print(f"矩阵的秩={jac_det}")
    pass
        
if __name__ == "__main__":
    robot = SR4()
    # test_fix(robot)
    test_jacob(robot)