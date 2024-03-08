import numpy as np
import math
from math import pi
from matplotlib import pyplot as plt

R2D = 180/pi
D2R = pi/180

def calculate_parabola_param(s1,s2,s3,y1,y2,y3):
    c = y1
    a = (y2-c-(y3-c)*(s2/s3))/(s2*s2-s2*s3)
    b = (y3-c-a*s3*s3)/s3
    return a,b,c
    
    
def test_point():
    js = np.array([-15,0,0,15,60,-20])
    jm = np.array([5,0,0,5,20,10])
    je = np.array([20,0,0,15,30,10])
    j1 = np.array([-15,0,0,15,-30,-20])
    j2 = np.array([0,0,0,0,0,-25])
    s1,s2,s3 = 0,15,35
    # a1,b1,c1 = calculate_parabola_param(s1,s2,s3,js[3],jm[3],je[3])
    # a2,b2,c2 = calculate_parabola_param(s1,s2,s3,js[4],jm[4],je[4])
    # a3,b3,c3 = calculate_parabola_param(s1,s2,s3,js[5],jm[5],je[5])
    a1,b1,c1 = calculate_parabola_param(s1,s2,s3,j1[3],j2[3],je[3])
    a2,b2,c2 = calculate_parabola_param(s1,s2,s3,j1[4],j2[4],je[4])
    a3,b3,c3 = calculate_parabola_param(s1,s2,s3,j1[5],j2[5],je[5])
    # print(f"j4 param={a1,b1,c1}")
    # print(f"j5 param={a2,b2,c2}")
    # print(f"j6 param={a3,b3,c3}")
    j4a = interpolate(a1,b1,c1,s1,s2)
    j4b = interpolate(a1,b1,c1,s2,s3)
    j4 = j4a+j4b
    j5a = interpolate(a2,b2,c2,s1,s2)
    j5b = interpolate(a2,b2,c2,s2,s3)
    j5 = j5a+j5b    
    j6a = interpolate(a3,b3,c3,s1,s2)
    j6b = interpolate(a3,b3,c3,s2,s3)
    j6 = j6a+j6b
    t = np.linspace(s1,s3,s3*10)
    fig=plt.figure()
    plt.plot(t,j4,'r',label='joint4')
    plt.plot(t,j5,'b',label='joint5')
    plt.plot(t,j6,'y',label='joint6')
    plt.legend()
    plt.show()
    
    
    
def interpolate(a,b,c,s1,s2):
    t = np.linspace(s1,s2,(s2-s1)*10)
    joint = []
    for i in range(len(t)):
        y = a*t[i]*t[i]+b*t[i]+c
        joint.append(y)
    return joint
    
if __name__ == "__main__":
    test_point()