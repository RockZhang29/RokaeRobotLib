import numpy as np
import math
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

pi = np.pi
R2D = 180/pi
D2R = pi/180
tool_frame = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.14],[0,0,0,1]])

def rotm2eular(R) :
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
        
def eular2rotm(eular):
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
       
def rotm2quat(RT):
    r = R.from_matrix(RT)
    return r.as_quat()
 
def rotm2quat2(RT):
    q = Quaternion(matrix=RT)
    quat = np.array([q.x,q.y,q.z,q.w])
    return quat

def test():
    # q1 = np.array([-0.20204424,  0.90065981, -0.3831929 , -0.03395916])
    # q2 = np.array([ 0.09977044,  0.91713932, -0.37227702,  0.10154385])
    # q3 = np.array([-0.90060463,  0.06324738, -0.09885098, -0.41849679])
    # q4 = np.array([ 0.85585154, -0.02477228,  0.0875352 ,  0.50915819])
    # q5 = np.array([-0.00287278, -0.81462392,  0.57787853, -0.04935603])
    #rq = np.array([[-0.95946941,0.25861168, 0.11197523],[ 0.10740191,  0.70291136 ,-0.70312193],[-0.2605442, -0.66259763, -0.70219733]])
   # r1 = R.from_matrix(np.array([[-0.95946941,0.25861168, 0.11197523],[ 0.10740191,  0.70291136 ,-0.70312193],[-0.2605442, -0.66259763, -0.70219733]]))
    # e1 = r1.as_euler('xyz', degrees=False)
    # print(f"quad={rotm2quat2(rq)}")
    # print(f"quat={r1.as_quat()}")

    j1 = np.array([-1.77799,-0.199206,-2.61334,-0.399962,0.653154,1.84284])
    jnt = j1
    for i in range(len(jnt)):
        jnt[i] = np.arctan2(np.sin(jnt[i]),np.cos(jnt[i]))
    ed = np.array([102.764,-0.186,2.128])*D2R
    
    print(f"jnt={jnt*R2D}")
    ex = R.from_euler('xyz',ed,degrees=False)
    qb = ex.as_quat()
    print(f"ed={ed}")
    print(f"qb={qb}")
    q1 = rotm2quat(eular2rotm(np.array([2.27535,-0.233094,-0.0320564])))
    q2 = rotm2quat(eular2rotm(np.array([2.06406,-0.175967,0.0474866])))
    # q3 = rotm2quat(eular2rotm(np.array([2.36033,0.221613,0.350828])))
    # print(q1,q2,q3)
    
    key_times = [0,19]
    q = [q1,q2]
    times = np.linspace(0,19,20)
    key_rots = R.from_quat(q)

    slerp = Slerp(key_times, key_rots)
    interp_rots = slerp(times)
    quad = interp_rots.as_quat()
    # print(f"quad ={quad}")
    plt.plot(times,quad)
    # plt.show()
    
    

if __name__ == "__main__":
    test()