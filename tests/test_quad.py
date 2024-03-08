import math
import numpy as np
import roboticstoolbox as rtb

pi = np.pi
R2D = 180/pi
D2R = pi/180
sin = np.sin
cos = np.cos

def test_quad():
    a = np.array([1,0,0,0])
    b = np.array([0,0,1,0])
    theta = 45*R2D
    c = a*cos(theta)+b*sin(theta)
    print(f"c={c}")
    m = c[0]**2+c[2]**2
    print(f"总和={m}")
    
if __name__ == "__main__":
    test_quad()
