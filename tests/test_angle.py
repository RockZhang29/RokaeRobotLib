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

pi = np.pi
R2D = 180/pi
D2R = pi/180

def test():
    j1 = np.array([1.79395,0.0873863,1.84406,0.0533822,0.488315,-0.503091])*R2D
    j2 = np.array([2.27531,-0.222625,1.8838,-1.09897,0.766817,0.920442])*R2D
    print(f"j1={j1}")
    print(f"j2={j2}")
    
if __name__ == "__main__":
    test()