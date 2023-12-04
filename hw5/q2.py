import numpy as np

CDE = 1e-3
U = 5
rho = 1.2
cp = 1005
Lv = 2.501e6
Rv = 461.5

Be = lambda Ts,qss: Lv/cp * (qss * dqdT(qss,Ts))
qsa = lambda Ts,Ta,qss: qss + (Ta-Ts)*qss*(Lv/(Rv*Ts**2))

def dqdT(qstar,T):
    return qstar * L / (Rv*T**2)

def q2(RH,Ts,Ta,qss):
    #LE = rho*Lv*CDE*U*( (1-RH)*qss + RH*cp/Lv*(Ts-Ta)/Be(Ts,qss) )
    LE = rho * Lv * CDE * U * ( (1-RH)*qss + RH*cp*(Ts-Ta)/Lv/Be(Ts,qss) )
    Bo = cp*(Ts-Ta)/LE

def q3(RH,Ts,Ta,qss):
    LE = rho*Lv*CDE*U*( (1-RH)*qss + RH*cp/Lv*(Ts-Ta)/Be(Ts,qss) )
    Bo = cp*(Ts-Ta)/LE
    print(Bo)


if __name__=="__main__":
    q2_args = (
            (.5, 273.15, 271.15, 3.75e-3),
            (1, 273.15, 271.15, 3.75e-3),
            (.5, 303.15, 301.15, 27e-3),
            (1, 303.15, 301.15, 27e-3),
            )
    #q2(RH=.5,Ts=273.15,Ta=271.15,qss=)
    for a in q2_args:
        q2(*a)


    Ea = rho*CDE*U*qsa(Ts,Ta,qss)*(1-RH) #* 1e3 * (60*60*24) #* Be(Ts,qss) / (1+Be(Ts,qss))
    print(Ea)

    dqdT = (qss-qsa(Ts,Ta,qss))/(Ts-Ta)

    """ """

    RH = .7
    dT = 2
    Ts = np.array([0,15,30]) + 273.15
    qss = np.array([4,10,25]) * 1e-3
    Ta = Ts + 2

