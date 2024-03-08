import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


pi = np.pi
R2D = 180/pi
D2R = pi/180


def read():
    # path = 'C:\\Users\\13651\\Desktop\\Data.xlsx'
    # data = pd.DataFrame(pd.read_excel(path))
    path1 = 'C:\\Users\\13651\\Desktop\\SingData.xlsx'
    path2 = 'C:\\Users\\13651\\Desktop\\NormData.xlsx'
    # df_list = pd.read_excel(path, sheet_name=7)
    singdata = pd.read_excel(path1)
    normdata = pd.read_excel(path2)
    js1 = singdata['J1']*R2D
    js2 = singdata['J2']*R2D
    js3 = singdata['J3']*R2D
    js4 = singdata['J4']*R2D
    js5 = singdata['J5']*R2D
    js6 = singdata['J6']*R2D
    jn1 = normdata['J1']*R2D
    jn2 = normdata['J2']*R2D
    jn3 = normdata['J3']*R2D
    jn4 = normdata['J4']*R2D
    jn5 = normdata['J5']*R2D
    jn6 = normdata['J6']*R2D
    plt.title('Sing Data Joint4-6')
    s1 = np.linspace(0,1,len(js1))
    s2 = np.linspace(0,1,len(jn1))
    plt.ylabel('deg')
    # plt.plot(s1,js1,label="sing J1")
    # plt.plot(s1,js2,label="sing J2")
    # plt.plot(s1,js3,label="sing J3")
    plt.plot(s1,js4,label="sing J4")
    plt.plot(s1,js5,label="sing J5")
    plt.plot(s1,js6,label="sing J6")
    # plt.plot(s2,jn1,label="norm J1")
    # plt.plot(s2,jn2,label="norm J2")
    # plt.plot(s2,jn3,label="norm J3")
    plt.plot(s2,jn4,label="norm J4")
    plt.plot(s2,jn5,label="norm J5")
    plt.plot(s2,jn6,label="norm J6")
    plt.legend()
    plt.grid()
    plt.show()

def test():
    a1,b1,c1,d1 = -2.18117,3.95922,-2.45178,0.261799
    a2,b2,c2,d2 = 1.57307,-3.83072,-1.12284,1.5708
    a3,b3,c3,d3 = 3.133,-2.31812,0.101786,0
    t = np.linspace(0,0.693026,300)
    j4,j5,j6 = 0,0,0
    x,y,z = [],[],[]
    for i in t:
        j4 = a1*i**3+b1*i**2+c1*i+d1
        j5 = a2*i**3+b2*i**2+c2*i+d2
        j6 = a3*i**3+b3*i**2+c3*i+d3
        x.append(j4)
        y.append(j5)
        z.append(j6)
    plt.figure()
    plt.xlabel('s')
    plt.ylabel('deg')
    plt.plot(t,x,label="J4")
    plt.plot(t,y,label="J5")
    plt.plot(t,z,label="J6")
    plt.legend()
    plt.grid()
    plt.show()

 
if __name__ == "__main__":
    # test()
    read()