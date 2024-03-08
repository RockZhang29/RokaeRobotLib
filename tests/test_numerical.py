import numpy as np
import math
import random
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
import time

pi = np.pi
R2D = 180/pi
D2R = pi/180
tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])

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
                
class TestSR4:
    def __init__(self,robot):
        self.robot = robot
        self.d1 = 0.355
        self.dx = 0.4
        self.dy = -0.05
        self.tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
        
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
        T0 = SE3(np.dot(self.set_jnt_angle(start_jnt),self.tool_frame))
        T1 = SE3(np.dot(self.set_jnt_angle(end_jnt),self.tool_frame))
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint_j = TJ.s.tolist() # 得到456轴
        TL = tr.ctraj(T0,T1,t) # tcp MoveL的直线
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
            if i == 0:
                joint_index = start_jnt
                # joint_list.append(start_jnt)
            else:
                joint_index = joint_list[i-1]
            j = self.ink_a(joint,x,y,z,joint_index) 
            fk = robot.fkine(j)
            pk = self.get_position(fk)
            xa = self.get_position(TL[i])[0]
            ya = self.get_position(TL[i])[1]
            za = self.get_position(TL[i])[2]
            if abs(pk[0]-xa) > 1e-04 or abs(pk[1]-ya) > 1e-04 or abs(pk[2]-za) > 1e-04:
                print(f"位置={pk[0],xa}")
            joint_list.append(j)
            # fk = SE3(np.dot(robot.fkine(j),self.tool_frame))
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
    
    def moveAbsJ(self,start_jnt,end_jnt,t):
        TJ = tr.jtraj(start_jnt,end_jnt,t)
        joint = TJ.s.tolist()
        # robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
        return joint
       
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
            print(f"数组长度错误")
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
            pk = self.get_position(fk)
            if abs(pk[0]-x) > 1e-04 or abs(pk[1]-y) > 1e-04 or abs(pk[2]-z) > 1e-04:
                print(f"位置不对")
                return False
            output_joint.append(j)
        return output_joint
    
    def space_line(self,p1,p2,t):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dz = p2[2]-p1[2]
        d = np.array([dx,dy,dz])
        point = p1+0.05*t*d
        return point
        
    def moveL(self,start_jnt,end_jnt,t):
        Ts = SE3(np.dot(self.set_jnt_angle(start_jnt),self.tool_frame))
        Te = SE3(np.dot(self.set_jnt_angle(end_jnt),self.tool_frame))
        jnt = (end_jnt-start_jnt)/len(t)
        # print(f"起始点={ps}")
        TL = tr.ctraj(Ts,Te,t)
        joint=[]
        for i in range(0,len(TL)):
            # p = self.get_position(TL[i])
            # print(TL[i])
            if i == 0:
                j = start_jnt
            else:
                # 在这里进行多次求数值解q0，看看符合不符合
                j = self.get_joint(TL[i],start_jnt+i*jnt)
            joint.append(j)
        return joint,TL
        
def test_ik_LMS(robot):
    t = np.linspace(0,1,10) 
    Robot = TestSR4(robot)
    qs = np.array([-1.6572,-0.26248,-2.41557,0.640445,-0.260787,0.482477])
    qe = np.array([-1.56714,-0.299403,-2.56364,1.06237,-0.134317,0.139304])
    Ts = robot.fkine(qs)
    Te = robot.fkine(qe)
    print(f"start_joint={qs*R2D}")

    # print(f"end_joint={qe*R2D}")
    jnt1 = robot.ikine_LMS(Ts,qs).q
    jnt2 = robot.ikine_LMS(Te,qe).q
    # print(f"start_ans={jnt1*R2D}")
    # print(f"end_ans={jnt2*R2D}")
    
    jl,TL = Robot.moveL(qs,qe,t)
    # print(f"jl={jl}")
    # jm = Robot.replan_movetype(qs,qe,t)
    Ttest = TL[1]
    print(f"TL={TL}")
    d = qe-qs
    # print(f"差距={d*10}")
    joint = []
    stime = time.time()
    for i in range(2):
        # 生成合适的对比q
        jnt = np.array([0,0,0,0,0,0])
        for j in range(6):
            if d[j]>=0:
                jnt[j] = random.uniform(0,d[j])
            else:
                jnt[j] = random.uniform(d[j],0)
        qi = qs+jnt
        # 数值解逆解
        jnt_test = robot.ikine_LMS(Ttest,q0=qi).q
        joint.append(tuple(jnt_test))
    # 列表去重    
    joint = list(set(joint))
    for i in range(len(joint)):
        joint[i] = np.array(list(joint[i]))*R2D
    etime = time.time()
    #print("程序运行时间：%.2f秒" % (etime - stime))
    # 打印所有
    # print(f"joint={joint}")
    # 选择最近的
    # xplot(np.array(jl)*R2D,block=True)
    
def test_fk(robot):
    Robot = TestSR4(robot)
    tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
    
    j2_list = np.array([[-0.9240879473,  2.005092454,  1.853288743,   2.681439731,  2.036318251, -1.791672559],
                        [ -1.739832339,   1.98786561,  1.903183446,  0.2980935161,  -2.09023542,  1.238345314],
                        [  2.203812096, -2.208578609, -2.304912315, -0.5228834829,  2.255305665, -1.931378874],
                        [  1.409285584, -2.205439692, -2.367683234,  -2.792966731, -2.324672137,  1.335369389]])
    # 0.09176,-0.3919,0.495
    for i in range(len(j2_list)):
        T = robot.fkine(j2_list[i])
        # print(f"T{i}={T}")
        
    j1 = np.array([-1.06607,-0.251831,-2.51718,-4.31554,0.383044,5.81831])
    p1 = Robot.get_position(SE3(np.dot(robot.fkine(j1),tool_frame)))
    j3 = np.array([-1.1675,-0.251763,-2.55738,-2.81371,-0.389104,4.32218])
    p3 = Robot.get_position(SE3(np.dot(robot.fkine(j3),tool_frame)))
    print(f"p1={p1}")
    print(f"p2={[0.108464,-0.506244,0.415898]}")    
    print(f"p3={p3}")

if __name__ == "__main__":  # pragma nocover
    robot = SR4()
    test_ik_LMS(robot)
    # test_fk(robot) 