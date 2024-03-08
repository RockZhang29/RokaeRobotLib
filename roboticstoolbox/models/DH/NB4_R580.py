from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH 

class NB4_R580(DHRobot):
    def __init__(self):
        deg = pi / 180
        
        # robot length values (metres)
        a1 = 0.333
        a2 = 0.280
        a3 = 0.015
        d4 = 0.300
        d6 = 0.068
        
        L0 = RevoluteDH(
            d=0,          # link length (Dennavit-Hartenberg notation)
            a=0,          # link offset (Dennavit-Hartenberg notation)
            alpha=0,   # link twist (Dennavit-Hartenberg notation)
            qlim=[-171*deg, 171*deg],
            )    # minimum and maximum joint angle
        L1 = RevoluteDH(
            d=0,
            a=a1,
            alpha=0,
            qlim=[-90*deg,130*deg],
        )
        L2 = RevoluteDH(
            d=0,
            a=a2,
            alpha=0,
            qlim=[-200*deg,55*deg],
        )
        L3 = RevoluteDH(
            d=d4,
            a=a3,
            alpha=0,
            qlim=[-171*deg,171*deg],
        )
        L4 = RevoluteDH(
            a=0,
            d=0,
            alpha=0,
            qlim=[-120*deg,120*deg],
        )
        L5 = RevoluteDH(
            a=0,
            d=d6,
            alpha=pi/2,
            qlim=[-360*deg,360*deg],
        )

        super().__init__(
            [L0, L1, L2, L3, L4, L5],
            name="NB4_R580",
            manufacturer="Rokae",
            )
        
        self.qr = np.array([-30*deg, 0, 0, 0, 45 * deg, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.addconfiguration_attr(
            "qd", np.array([0, -90 * deg, 180 * deg, 0, 0, -90 * deg])
        )
        
if __name__ == "__main__":  # pragma nocover

    robot = NB4_R580()
    print(robot)