import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import tr2x, numjac, numhess
from scipy.linalg import block_diag
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

class CR7(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.296)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.490)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        l3 = Link(ET.tz(0.360)*ET.Rz(), name="link3", parent=l2)
        l4 = Link(ET.ty(0.150)*ET.Ry(), name="link4", parent=l3)
        # l5 = Link(ET.tz(0.127)*ET.Rz(), name="link5", parent=l4)
        l5 = Link(ET.tz(0.127)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
        elinks = [l0,l1,l2,l3,l4,l5]
        super(CR7, self).__init__(elinks, name="CR7", manufacturer="Rokae")  

class CR7f(ERobot):
    def __init__(self):
        l0 = Link(ET.Rz(), name="link0", parent=None)
        l1 = Link(ET.tz(0.296)*ET.Ry(), name="link1", parent=l0)
        l2 = Link(ET.tz(0.490)*ET.Rz(pi)*ET.Ry(), name="link2", parent=l1)
        elinks = [l0,l1,l2]
        super(CR7f, self).__init__(elinks, name="CR7f", manufacturer="Rokae")  

class CR7b(ERobot):
    def __init__(self):
        l3 = Link(ET.tz(0.360)*ET.Rz(), name="link3", parent=None)
        l4 = Link(ET.ty(0.150)*ET.Ry(), name="link4", parent=l3)
        l5 = Link(ET.tz(0.127)*ET.Rz(pi)*ET.Rz(), name="link5", parent=l4)
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
        self.tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
    
    def cal_DH(self,joint):
        cos = np.cos
        sin = np.sin
        j1 = joint[0]
        j2 = joint[1]
        j3 = joint[2]
        j4 = joint[3]
        j5 = joint[4]
        j6 = joint[5]
        T01 = np.mat([[cos(j1),-sin(j1),0,0],[sin(j1),cos(j1),0,0],[0,0,1,0],[0,0,0,1]])
        T12 = np.mat([[cos(j2),0,sin(j2),0],[0,1,0,0],[-sin(j2),0,cos(j2),self.d1],[0,0,0,1]])
        T23 = np.mat([[-cos(j3),0,-sin(j3),0],[0,-1,0,0],[-sin(j3),0,cos(j3),self.a2],[0,0,0,1]])
        T34 = np.mat([[cos(j4),-sin(j4),0,0],[sin(j4),cos(j4),0,0],[0,0,1,self.d4],[0,0,0,1]])
        T45 = np.mat([[cos(j5),0,sin(j5),0],[0,1,0,self.d5],[-sin(j5),0,cos(j5),0],[0,0,0,1]])
        T56 = np.mat([[-cos(j6),sin(j6),0,0],[-sin(j6),-cos(j6),0,0],[0,0,1,self.d6],[0,0,0,1]])
        T02 = np.dot(T01,T12)
        T03 = np.dot(T02,T23)
        T04 = np.dot(T03,T34)
        T05 = np.dot(T04,T45)
        T06 = np.dot(T05,T56)
        return T01,T02,T03,T04,T05,T06
       
     
def test_link(robot):
    Robot = TestCR7(robot)    
    rob1 = CR7f()
    rob2 = CR7b()
    # Robot.test_jacob0()  
    j1 = np.array([0,0,pi/2,0,pi/4,0])
    # j1 = np.array([-55.449,38.440,-116.095,-31.074,-10.298,62.921])*D2R
    fk = robot.fkine(j1)
    # t06 = Robot.cal_DH(j1)
    #print(f"t06={t06}")
    # tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])
    fkf = rob1.fkine(j1[0:2])
    fkb = rob2.fkine(j1[3::])
    #print(f"fkf={fkf}")
    #print(f"fkb={fkb}")
    fkc = np.dot(fkf,fkb)
    #print(f"fk={fk}")
    # print(f"fkc={fkc}")
    
def test_jac(robot):
    Robot = TestCR7(robot)  
    j1 = np.array([0,0,pi/2,0,pi/4,0])
    T = robot.fkine(j1)
    T01,T02,T03,T04,T05,T06 = Robot.cal_DH(j1)
    R01 = T01[0:3,0:3]
    R02 = T02[0:3,0:3]
    R03 = T03[0:3,0:3]
    R04 = T04[0:3,0:3]
    R05 = T05[0:3,0:3]
    R06 = T06[0:3,0:3]
    z01 = T01[3,0:3]
    z02 = T02[3,0:3]
    z03 = T03[3,0:3]
    z04 = T04[3,0:3]
    z05 = T05[3,0:3]
    z06 = T06[3,0:3]
        
if __name__ == "__main__":
    # robot = CR7()
    # test_jac(robot)
    x = 0.97385
    a2 = math.asin(x)
    a3 = np.pi-a2
    print(f"{a2},{a3}")
    b2 = -0.734927
    q2 = a2-b2
    q3 = a3-b2
    print(f"{q2},{q3}")
    print(f"{(q2-np.pi)*R2D},{(q3-np.pi)*R2D}")
