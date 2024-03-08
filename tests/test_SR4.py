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
cos = np.cos
sin = np.sin
tool_frame = np.array([[1,0,0,-0.005],[0,1,0,0.01],[0,0,1,0.14],[0,0,0,1]])


class SR4t(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.355)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.400)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        # SR3
        # l1 = Link(ET.tz(0.313)*ET.Ry(), name="link1", parent=l0)
        # l2 = Link(ET.tz(0.290)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        elinks = [l0,l1,l2]
        super(SR4t, self).__init__(elinks, name="SR4t", manufacturer="Rokae") 
    
class SR4b(ERobot):  
    def __init__(self):  
        # SR4
        l3 = Link(ET.tz(0.400)*ET.tx(0.05)*ET.Rz(), name="link3")
        l4 = Link(ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.ty(-0.136)*ET.tz(0.1035)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
        # SR3
        # l3 = Link(ET.tz(0.290)*ET.tx(0.05)*ET.Rz(), name="link3", parent=None)
        # l4 = Link(ET.ty(-0.136)*ET.Ry(), name="link4", parent=l3)
        # l5 = Link(ET.tz(0.1035)*ET.Rz(), name="link5", parent=l4)
        elinks = [l3,l4,l5]
        super(SR4b, self).__init__(elinks, name="SR4b", manufacturer="Rokae")  
        
class SR4(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.355)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.400)*ET.Rz(pi)*ET.tx(-0.05)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.400)*ET.tx(0.05)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.ty(-0.136)*ET.tz(0.1035)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(SR4, self).__init__(elinks, name="SR4", manufacturer="Rokae")  

class TestSR4:
    def __init__(self,robot):
        self.robot = robot
        # self.d1 = 0.328 # SR4-C
        self.d1 = 0.355 # SR4
        self.dx = 0.4 # SR4
        # self.d1 = 0.313 # SR3
        # self.dx = 0.290 # SR3
        self.dy = -0.05
        self.tool_frame = np.array([[1,0,0,-0.005],[0,1,0,0.01],[0,0,1,0.14],[0,0,0,1]])
        #self.tool_frame = np.array([[1,0,0,-0.02594],[0,1,0,0.00102],[0,0,1,0.3598],[0,0,0,1]])
        # self.tool_frame = np.array([[1,0,0,-0.02594],[0,1,0,0.00102],[0,0,1,0.3598],[0,0,0,1]])
        
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
    
    def eular2quat(self,eular):
        r = R.from_euler(eular)
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
        T36 = rob_back.fkine(joints[3::])
        # print(f"j4-j6={joints[3::]}")
        # print(f"T36={T36}")
        T3t = SE3(np.dot(T36,self.tool_frame))
        # print(f"T3t={T3t}")
        pose = self.get_position(T3t)
        Px,Py,Pz = pose[0],pose[1],pose[2]
        print(f"输入的笛卡尔位置={[x,y,z]}")
        print(f"需要的T3t位置={Px,Py,Pz}")
        # 计算joint1
        if abs(Py/np.sqrt(x**2+y**2)) > 1:
            print(f"计算j1出现不可解的数学问题,{Py,np.sqrt(x**2+y**2)}")
            return False
        else:
            a1 = np.arcsin(Py/np.sqrt(x**2+y**2))
            b1 = np.arctan2(-y,x)
            print(f"a1,b1={a1,b1}")
            theta1 = self.second_sort(a1,b1,j1)
        # print(f"theta1={theta1}")
        # 计算joint2
        m1 = (x-sin(theta1)*Py)/cos(theta1)
        m2 = z - self.d1
        ind1 = m1*self.dx+m2*self.dy #dx是l23，dy是-l23x
        ind2 = m2*self.dx-m1*self.dy
        n = (m1**2+m2**2+self.dx**2+self.dy**2-Px**2-Pz**2)/(2)
        # print(f"m1={m1},m2={m2},n={n},ind1={ind1},ind2={ind2}")
        if abs(n/np.sqrt(ind1**2+ind2**2)) > 1:
            print(f"计算j2出现不可解的数学问题,{n,np.sqrt(ind1**2+ind2**2)}")
            return False
        else:
            a2 = np.arcsin(n/np.sqrt(ind1**2+ind2**2))
            b2 = np.arctan2(ind2,ind1)
            # print(f"a2={a2},b2={b2}")
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
            # print(f"a3={a3},b3={b3},k={k}")
            theta23 = self.second_sort(a3,b3,j2-j3)
            theta3 = theta2-theta23
            # print(f"j2-j3={(j2-j3)*R2D}")
            # print(f"theta23={theta23*R2D}")
        # 输出
        joint = [theta1,theta2,theta3,j4,j5,j6]
        if self.judge_position(x,y,z,joint,joints) is True:
            # print(f"输出的joint={joint}")
            return joint
        else:
            # print(f"错误的joint={jnt}")
            eps23 = self.get_another(a3,b3,j2-j3)
            eps3 = theta2-eps23
            joint = [theta1,theta2,eps3,j4,j5,j6]
            # print(f"eps23={eps23*R2D},eps3={eps3*R2D}")
            if self.judge_position(x,y,z,joint,joints) is True:
                # print(f"解析正确!")
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
        # for i in range(len(TL)):
        #     print(f"第{i}个TL={TL[i]}")
        # pose = self.line(start_jnt,end_jnt,t)
        # for i in range(len(TL)):
        #     print(f"第{i}个={TL[i]}")
        joint_list = []
        for i in range(0,len(t)):
            joint = joint_j[i]
            x = self.get_position(TL[i])[0]
            y = self.get_position(TL[i])[1]
            z = self.get_position(TL[i])[2]
            # print(f"第{i+1}条路径MoveL提供的位置{x,y,z}")
            # print(f"第{i}条路径")
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
                joint_list.append(j)
                
        joint_list = joint_list[:-1]
        joint_list.append(end_jnt)
        # robot.plot(np.array(joint_list),backend='pyplot',dt=0.01,block=True)
        # self.show(xl,yl,zl,joint_list)
        return joint_list
     
    def line(self,start_jnt,end_jnt,t):
        fk1 = SE3(robot.fkine(start_jnt)*self.tool_frame)
        fk2 = SE3(robot.fkine(end_jnt)*self.tool_frame)
        p1 = self.get_position(fk1)
        p2 = self.get_position(fk2)
        dis_x = p2[0]-p1[0]
        dis_y = p2[1]-p1[1]
        dis_z = p2[2]-p1[2]   
        p = []
        for i in range(len(t)):
            pos = np.array([p1[0]+i*dis_x/len(t),p1[1]+i*dis_y/len(t),p1[2]+i*dis_z/len(t)])
            p.append(pos)
            print(pos)
        return p
    
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
        # print(f"joint1,joint2={joint1,joint2}")
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
                js[i] = js[i]
                joints[i] = joints[i]
            jp = np.array(js)*R2D
            jop = np.array(joints)*R2D
            print(f"输入角度:{jop}")
            print(f"新解角度:{jp}")
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
        # 需要rad
        joint = [0,0,0,0,0,0]
        for i in range(len(jnt)):
            joint[i] = np.arctan2(np.sin(jnt[i]),np.cos(jnt[i]))
        return joint
        
def test_moveS(robot):
    t = np.linspace(0,1,10) 
    Robot = TestSR4(robot)
    j1 = np.array([-1.09797,-0.165406,-2.63374,-2.73708,-0.707649,4.93775])
    j2 = np.array([-1.34843,-0.19394,-2.72813,-3.37809,-0.620601,5.45177])
    joint = Robot.replan_movetype(j1,j2,t)
    xplot(np.array(joint)*R2D,block=True)
    robot.plot(np.array(joint),backend='pyplot',dt=0.01,block=True)
    print(joint)
    
def test_movel(robot):
    t = np.linspace(0,1,100) 
    Robot = TestSR4(robot)
    j1 = np.array([-1.09797,-0.165406,-2.63374,-2.73708,-0.707649,4.93775])
    j2 = np.array([-1.34843,-0.19394,-2.72813,-3.37809,-0.620601,5.45177])
    j1 = Robot.trans_in2_pi(j1)
    j2 = Robot.trans_in2_pi(j2)
    j,p = Robot.moveL(j1,j2,t)
    # print(f"笛卡尔角度={j1*R2D}")
    p1 = Robot.get_position(robot.fkine(j1))
    # print(f"终止角度={j2*R2D}")
    # p1 = Robot.get_position(SE3(np.dot(robot.fkine(j1),tool_frame)))
    # print(f"笛卡尔位置={p1}")
    # p2 = Robot.get_position(SE3(np.dot(robot.fkine(j2),tool_frame)))
    # print(f"终止位置={p2}")
    jm = Robot.replan_movetype(j1,j2,t)
    xplot(np.array(jm)*R2D,block=True)
    xplot(np.array(j)*R2D,block=True)
    for i in range(len(j)):
        j[i] = j[i]*R2D
    # print(f"moveL j={j}")
    # 对比MoveL姿态和MoveS姿态
    ang = []
    quads = []
    for i in range(0,len(jm)):
        jnt1 = jm[i]
        fk1 = robot.fkine(jnt1) 
        R1 = Robot.get_rotation(fk1)
        q = Robot.rotm2quat(R1)
        qs = q
        quads.append(qs)
        vector,angle = Robot.quat2axis(qs)
        ang.append(angle)
    # print(f"四元数={quads[0]}")
    
    
    q1 = Robot.rotm2quat(Robot.get_rotation(robot.fkine(j1)))
    q2 = Robot.rotm2quat(Robot.get_rotation(robot.fkine(j2)))
    
    quats = []
    q=[q1,q2]
    for i in range(len(q)-1):
        # if i == 0:
        key_times = [0,len(t)]
        times = np.linspace(0,len(t)-1,len(t))
        rots = R.from_quat([q[i],q[i+1]])
        slerp = Slerp(key_times, rots)
        interp_rots = slerp(times)
        # print(f"interp_rots{i}={interp_rots.as_quat()}")
        for j in range(len(interp_rots)):
            # quats.append(np.array(interp_rots[i].as_quat))
            a = interp_rots[j].as_quat()
            quats.append(a)
            # print(f"q{i}={a}")

    ane = []
    plt.title("Quads Path")
    plt.plot(t,quats,'b',label="Slerp")
    plt.plot(t,quads,'r',label="Wrist")
    plt.show()
    # print(f"SLERP的四元数={quats[-1]}")
    # print(f"插补的四元数={quads[-1]}")
    
    for i in range(len(quats)):
        v,ans = Robot.quat2axis(quats[i])
        ane.append(ans)
    # print(f"轴角={ang[0]}")
    
    # p1 = robot.fkine(j1)
    # ja = robot.ikine_LMS(p1).q
    # print(f"第二组关节角={ja*R2D}")
    # r1 = Robot.get_rotation(p1)
    # q1 = Robot.rotm2quat(r1)
    # v,a1 = Robot.quat2axis(q1)
    # print(f"四元数={q1}")
    # print(f"第二组轴角={a1}")
    plt.title("SR4 Fix Path Axis-Angle")
    plt.plot(t,ang,'r')
    plt.plot(t,ane,'b')
    plt.ylabel("deg")
    plt.show()
    
    dif = []
    plt.title("SR4 Fix Path Axis-Angle Difference")
    for i in range(len(ane)):
        d = abs(ane[i]-ang[i])
        dif.append(d)
    plt.plot(t,dif,'r')
    plt.ylabel("deg")
    plt.show()

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
    
    for i in range(len(quats)):
        v,ans = Robot.quat2axis(quats[i])
        ane.append(ans)
    plt.title("SR4 Fix Path Axis-Angle")
    plt.plot(te,ang,'r')
    plt.plot(te,ane,'b')
    plt.ylabel("deg")
    plt.show()
    
    dif = []
    plt.title("SR4 Fix Path Axis-Angle Difference")
    for i in range(len(ane)):
        d = abs(ane[i]-ang[i])
        dif.append(d)
    plt.plot(te,dif,'r')
    plt.ylabel("deg")
    plt.show()
        
def test_fixs(robot):
    t = np.linspace(0,1,500) 
    Robot = TestSR4(robot)
    # 实机数据组测试，首先获取joints和rpy，限位为[-2pi,2pi]
    # j1 = np.array([1.79347,0.0751479,1.81514,0.0423458,0.53441,-0.487623])
    # j2 = np.array([1.85765,0.0931603,2.17233,-0.962884,0.278414,0.545445])
    # j3 = np.array([1.79891,-0.024725,2.28979,-2.06926,0.550716,1.6677])
    # j4 = np.array([2.08105,-0.119716,2.18369,-1.6192,0.771304,1.41558])
    # p1 = np.array([0.0362844,-0.48985,0.420383]) # 1137
    
    # j1 = np.array([1.69824,0.02392,1.67774,0.207211,0.645741,-0.767597])
    # j2 = np.array([1.79395,0.0873863,1.84406,0.0533822,0.488315,-0.503091])
    # j3 = np.array([2.27531,-0.222625,1.8838,-1.09897,0.766817,0.920442])
    # j4 = np.array([2.29855,-0.318417,1.51388,-0.593093,1.00677,0.218777])
    # p1 = np.array([0.00840967,-0.492955,0.428268]) # 1613
    # p2 = np.array([0.036736,-0.486876,0.413651])
    # p3 = np.array([0.0666656,-0.490523,0.406308])
    # p4 = np.array([0.0965737,-0.502667,0.410112])
    # p5 = np.array([0.126286,-0.515947,0.414392])
    # p6 = np.array([0.155336,-0.528712,0.420956])
    
    # j1 = np.array([1.69771,0.0303006,1.6854,0.231309,0.653847,-0.755248])
    # j2 = np.array([1.85388,0.102029,1.94983,-0.192194,0.402126,-0.236948])
    # j3 = np.array([2.19497,-0.193086,2.03416,-1.42499,0.722544,1.23729])
    # j4 = np.array([2.29307,-0.237006,1.8171,-0.984126,0.799336,0.795651])
    # p1 = np.array([0.0154496,-0.490953,0.424911]) # 540
    # p2 = np.array([0.0440494,-0.486569,0.410337])
    # p3 = np.array([0.0742127,-0.493162,0.406905])
    # p4 = np.array([0.104007,-0.50602,0.411138])
    # p5 = np.array([0.133657,-0.519153,0.415721])
    
    # j1 = np.array([1.85501,0.0892247,1.90589,-0.178009,0.471948,-0.247207])
    # j2 = np.array([1.77784,0.0869142,2.23397,-1.29572,0.21419,0.840283])
    # j3 = np.array([1.91181,-0.0557056,2.26751,-1.88707,0.641471,1.56552])
    # j4 = np.array([2.08795,-0.134487,2.17392,-1.63773,0.778656,1.43674])
    # p1 = np.array([0.0435922,-0.489241,0.417344]) # 64
    
    # j1 = np.array([1.10253,-0.0292476,1.97277,-2.17959,-0.678016,1.23951])
    # j2 = np.array([1.15881,0.0166645,2.09819,-2.03803,-0.596212,1.21511])
    # j3 = np.array([1.64303,-0.023845,2.30599,-2.46314,0.539396,1.96355])
    # j4 = np.array([2.00522,-0.127766,2.24038,-1.84859,0.775478,1.60791])
    # p1 = np.array([0.00489231,-0.494073,0.429829]) # 3
    # p2 = np.array([0.033134,-0.487275,0.415516])
    
    # j1 = np.array([1.7253,0.0563838,1.74238,0.207873,0.594297,-0.6903])
    # j2 = np.array([1.81136,0.101277,2.17951,-0.865064,0.174226,0.412329])
    # j3 = np.array([1.93347,-0.109613,2.26994,-1.97992,0.74106,1.68862])
    # j4 = np.array([2.07877,-0.159452,2.18248,-1.73405,0.786437,1.52928])
    # p1 = np.array([0.0260169,-0.488457,0.419376]) # 1077
    # p2 = np.array([0.0552913,-0.48769,0.407024])
    
    j1 = np.array([1.73416,0.0459762,1.72761,0.168387,0.621479,-0.644108])
    j2 = np.array([1.89792,0.0976633,2.04674,-0.543323,0.377563,0.138557])
    j3 = np.array([2.01847,-0.0946308,2.22517,-1.72613,0.733414,1.48217])
    j4 = np.array([2.11005,-0.147414,2.14812,-1.60302,0.771821,1.40809])
    p1 = np.array([0.0256834,-0.49178,0.425604]) # 600
    p2 = np.array([0.0548533,-0.490164,0.414297])
    
    # 构造中间路径起点
    ps = Robot.get_position(SE3(np.dot(robot.fkine(j2),tool_frame)))
    pe = Robot.get_position(SE3(np.dot(robot.fkine(j3),tool_frame)))
    # 构造中间路径终点
    pnt = [ps,p1,p2,pe]
    path_a = Robot.replan_movetype(j1,j2,t)
    path_b = Robot.replan_movetype(j3,j4,t)
    
    t2 = np.linspace(0,1,(len(pnt)-1)*500) 
    # 获取轴空间路径
    joint_j = Robot.moveAbsJ(j2,j3,t2)
    joint_l = []
    for i in range(len(pnt)-1):
        a = len(t)*i
        b = len(t)*(i+1)
        joint_a = joint_j[a:b]
        joint_l.append(joint_a)
    # 获取笛卡尔空间路径
    points = []
    for j in range(len(pnt)-1):
        point_start = pnt[j]
        point_end = pnt[j+1]
        point = []
        for i in range(len(t)):
            # 如果t的长度改变，space_lint里面的参数也得变，不然就乱了
            pose = Robot.space_line(point_start,point_end,i)
            point.append(pose)
        points.append(point)
    
    # ax1 = plt.axes(projection='3d')
    # ax1.set_title("Cart Point In 3D Space")
    # for i in range(len(points)):
    #     path = points[i]
    #     point = pnt[i]
    #     ax1.plot3D(pnt[i][0],pnt[i][1],pnt[i][2],'r*')
    #     for j in range(len(points[i])):
    #         ax1.plot3D(path[j][0],path[j][1],path[j][2],'y.--')
    # ax1.plot3D(pnt[-1][0],pnt[-1][1],pnt[-1][2],'r*')
    # plt.show()
    
    joints = [] ## 仅测试纯奇异点路径的时候启用
    # joints = path_a
    for i in range(len(pnt)-1):
        path = Robot.replan2_move(joint_l[i],points[i])
        joints = joints+path
    # joints = joints+path_b
    te = np.linspace(0,1,len(joints)) 
    # print(joints)
    xplot(np.array(joints),block=True)
    # robot.plot(np.array(joints),backend='pyplot',dt=0.01,block=True)
    
    ## 提取关节角
            
    
    # 反解位置
    # ax1 = plt.axes(projection='3d')
    # ax1.set_title("Cart Point In 3D Space")
    for i in range(len(joints)):
        pos = Robot.get_position(SE3(np.dot(robot.fkine(joints[i]),tool_frame)))
        # if i%100==0:
        # print(f"{joints[i]}")
        # ax1.plot3D(pos[0],pos[1],pos[2],'r-*')
    # plt.show()
        
    # f = open('test.txt','w')
    # for i in range(len(joints)):
    #     f.write(str(joints[i])+'\n')
    # f.close()
    
    ## 计算矩阵的det
    det = []
    for i in range(len(joints)):
        Tj = robot.jacobe(joints[i])
        jac_det = np.linalg.matrix_rank(Tj)#返回矩阵的秩
        det.append(jac_det)
    t = np.linspace(0,1,len(joints))
    plt.title("Jacobe Rank")
    plt.plot(t,det,'r-*')
    plt.show()
        
def switch_pos(q1):
    a=q1[0]
    b=q1[1]
    c=q1[2]
    d=q1[3]
    q2=q1
    q2[0]=d
    q2[1]=a
    q2[2]=b
    q2[3]=c
    return q2          
   
def inverse_switch(q1):
    a = q1[0]
    b = q1[1]
    c = q1[2]
    d = q1[3]
    q2 = q1
    q2[0] = b
    q2[1] = c
    q2[2] = d
    q2[3] = a
    return q2
        
def test_SR4(robot):
    Robot = TestSR4(robot)
    j0 = np.array([35,-12,42,45,72,123])*D2R
    j1 = np.array([0,0,90,0,45,0])*D2R
    j2 = np.array([0,0,90,0,-45,0])*D2R
    j3 = np.array([0,0,90,0,0,0])*D2R
    t = np.linspace(0,1,100)
    fk = robot.fkine(j0)
    fkt = np.dot(fk,tool_frame)
    # print(f"fkt={fkt}")
    joints = Robot.replan_movetype(j1,j2,t)
    # j1 = np.array([0,0,0,0,0.72566,0])
    # x,y,z = -0.518928,0.146,0.543058
    # joint = Robot.ink_a(j1,x,y,z,j1)
    # print(joint)
    for i in range(len(joints)):
        print(joints[i])
    # joint = Robot.ink_a(start_jnt,pos[0],pos[1],pos[2],start_jnt)
    # print(f"逆解输出的角度={np.array(joint)}")
       
def test_melt(robot):
    Robot = TestSR4(robot)
    j1 = np.array([-30.723,39.636,-108.056,42.930,-3.497,-217.471])*D2R
    j2 = np.array([31.056,41.912,-99.583,32.306,-17.323,-225.479])*D2R
    t = np.linspace(0,1,100)
    path1 = Robot.replan_movetype(j1,j2,t)
    paths = path1
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
        
    # plt.title("SR3C Wrist Singular Fix Path RPY")
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
    q = [q1,q2]
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
            
    # 处理一下eular看着好点
    for i in range(len(euler)):
        if euler[i][2]<0:
            euler[i][2] = -1*euler[i][2]
        if E1[i][2]<0:
            E1[i][2] = -1*E1[i][2]
            
    difex = [] 
    difey = []
    difez = []
    for i in range(len(euler)):
        # print(f"第{i}个欧拉角={euler[i]},E1={E1[i]}")
        dif1 = abs(euler[i][0]-E1[i][0])
        dif2 = abs(euler[i][1]-E1[i][1])
        dif3 = abs(euler[i][2]-E1[i][2])
        difex.append(dif1)
        difey.append(dif2)
        difez.append(dif3)
        
    tq = np.linspace(0,1,len(euler))
    plt.title("Path Euler Comparision")
    plt.plot(te,euler,'b',label="Slerp")
    plt.plot(te,E1,'r',label="Wrist")
    plt.ylabel("deg")
    plt.show()
    
    plt.title("Path Euler Difference")
    plt.plot(te,difex,'r',label="euler_x")
    plt.plot(te,difey,'y',label="euler_y")
    plt.plot(te,difez,'b',label="euler_z")
    plt.legend()
    plt.ylabel("deg")
    plt.show()
        
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
        
    # plt.title("SR3C Melting Short Path Provided by Wanhao's Axis-Angle")
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
    ax.set_xlim3d(1.5, 3)
    ax.set_ylim3d(0, 0.5)
    ax.set_zlim3d(0, 0.5)
    plt.show()
    
    dif = []
    plt.title("SR3C Melting Short Path Provided by Wanhao's Axis-Angle Difference")
    for i in range(len(ane)):
        d = abs(ane[i]-ang[i])
        # if i == 120:
        print(f"第{i}个轴角,Slerp={ane[i]},Wrist={ang[i]}")
        print(f"第{i}个轴角的差值={d}")
        print(f"第{i}个欧拉角,Slerp={euler[i]},Wrist={E1[i]}")
        print(f"第{i}个欧拉角差距={difex[i],difey[i],difez[i]}")    
        print(f"第{i}个四元数,Slerp={quats[i]},Wrist={quads[i]}")
        dif.append(d)
    plt.plot(te,dif,'r')
    plt.ylabel("deg")
    plt.show()
    
def rot(robot):
    Robot = TestSR4(robot)
    j1 = np.array([-30.389,39.636,-108.361,44.618,-3.156,-218.882])*D2R
    j2 = np.array([31.056,41.912,-99.583,32.306,-17.323,-225.479])*D2R
    # tf = np.array([[0.59198742,-0.58326229,-0.5561978,-0.02594],[-0.25250403,-0.78959663,0.55926637,0.00102],[-0.7653709, -0.19063647,-0.61470735,0.3598],[0,0,0,1]])
    tf = np.array([[1,0,0,-0.02594],[0,1,0,0],[0,0,1,0.3598],[0,0,0,1]])
    T0 = robot.fkine(j1)
    T1 = robot.fkine(j2)
    t = np.linspace(0,1,100)
    # print(f"T0={T0}")
    fk0 = T0*tf
    fk1 = T1*tf
    TL = tr.ctraj(T0,T1,t) # tcp MoveL的直线 
    for i in range(len(TL)):
        print(f"第{i}个TL={TL[i]*tf}")
    # print(f"T1={T1}")
    ja = np.array([-30.24208979,39.67737911,-108.24902584,44.61787503,-3.1561438,-218.88206696])*D2R
    fa = robot.fkine(ja)*tf
    print(f"fa={fa}")
    joint = Robot.replan_movetype(j1,j2,t)
    # print(f"joint={joint}") #[0.7066476316075075, -0.28305574758642843, -0.05347427754188078]

def test_link():
    robo = SR4()
    robo1 = SR4t()
    robo2 = SR4b()
    jnt = np.array([0,0,90,0,45,0])*D2R
    jntt = jnt[0:2]
    jntb = jnt[3:5]
    fk = robo.fkine(jnt)
    j1,j2,j3 = jnt[0],jnt[1],jnt[2]
    j4,j5,j6 = jnt[3],jnt[4],jnt[5]

    fkt = robo1.fkine(jntt)
    fkb = robo2.fkine(jntb)
    
    T34 = np.mat([[cos(j4),-sin(j4),0,0.05],[sin(j4),cos(j4),0,0],[0,0,1,0.4],[0,0,0,1]])
    T45 = np.mat([[cos(j5),0,sin(j5),0],[0,1,0,0],[-sin(j5),0,cos(j5),0],[0,0,0,1]])
    T56 = np.mat([[-cos(j6),sin(j6),0,0],[-sin(j6),-cos(j6),0,-0.136],[0,0,1,0.1035],[0,0,0,1]])
    T35 = np.dot(T34,T45)
    T36 = np.dot(T35,T56)
    T3t = np.dot(T36,tool_frame)
    print(f"T3t={T3t}")
    
    # T0t = fk*tool_frame
    # T3t = fkb*tool_frame
    # print(f"T36={fkb}")
    # print(f"T3t={T3t}")
    # L0t = fkt*T3t
    # print(f"T0t={T0t}")
    # print(f"L0t={L0t}")
    

if __name__ == "__main__":
    robot = SR4()
    test_SR4(robot)
    # test_melt(robot)
    # test_fix(robot)
    # test_fixs(robot)
    # rot(robot)
    test_link()